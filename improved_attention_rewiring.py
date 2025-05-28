import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_sparse
from torch_sparse import SparseTensor, set_diag
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch_geometric.utils import coalesce, homophily
import copy
import numpy as np

class ImprovedAttentionRewirer:
    """改進的注意力重連器，專注於提高同質性和準確性"""
    
    def __init__(self, top_k=8, temperature=0.5, min_deg=5, min_deg_ratio=1.5, 
                 homophily_weight=2.0, feature_dropout=0.2, edge_dropout=0.1):
        super(ImprovedAttentionRewirer, self).__init__()
        self.top_k = top_k
        self.temperature = temperature  # 降低溫度參數，使注意力更加集中
        self.min_deg = min_deg  # 增加最小度閾值
        self.min_deg_ratio = min_deg_ratio  # 增加最小度比例
        self.homophily_weight = homophily_weight  # 同質性權重
        self.feature_dropout = feature_dropout  # 特徵丟棄率，用於增強模型泛化能力
        self.edge_dropout = edge_dropout  # 邊丟棄率，用於防止過擬合
        
    def forward(self, x, edge_index=None, labels=None):
        """
        使用改進的注意力機制重新連接圖結構，同時增強同質性
        
        Args:
            x: 節點特徵矩陣 [num_nodes, feat_dim]
            edge_index: 原始邊索引 [2, num_edges]
            labels: 節點標籤 [num_nodes]
            
        Returns:
            new_edge_index: 重新連接後的邊索引 [2, num_new_edges]
        """
        num_nodes = x.size(0)
        device = x.device
        
        # 特徵預處理：應用丟棄和歸一化以增強泛化能力
        if self.training:
            x = F.dropout(x, p=self.feature_dropout, training=self.training)
        
        # 計算查詢和鍵特徵，使用L2歸一化提高穩定性
        x_norm = F.normalize(x, p=2, dim=1)  # L2歸一化特徵
        q_features = x_norm
        k_features = x_norm
        
        # 計算注意力分數矩陣 (使用批處理以節省內存)
        batch_size = 128
        attention_scores = torch.zeros(num_nodes, num_nodes, device=device)
        
        for i in range(0, num_nodes, batch_size):
            end_idx = min(i + batch_size, num_nodes)
            batch_q = q_features[i:end_idx]
            
            # 計算批次的注意力分數
            batch_scores = torch.mm(batch_q, k_features.t()) / self.temperature
            attention_scores[i:end_idx] = batch_scores
        
        # 創建新的邊索引
        new_edge_index = []
        
        # 對每個節點，選擇top-k個最相似的節點建立連接
        for node in range(num_nodes):
            # 獲取該節點的注意力分數
            node_similarity = attention_scores[node].clone()  # 創建副本以避免修改原始張量
            
            # 將自身的分數設為最小，避免自環
            node_similarity[node] = float('-inf')
            
            # 選擇top-k個最相似的節點
            _, top_indices = torch.topk(node_similarity, self.top_k)
            
            # 創建新的邊
            src = torch.full((self.top_k,), node, device=device)
            dst = top_indices
            
            # 添加到邊列表
            new_edge_index.append(torch.stack([src, dst]))
        
        # 合併所有邊
        new_edge_index = torch.cat(new_edge_index, dim=1)
        
        # 確保每個節點的最小度
        in_degree = torch.zeros(num_nodes, device=device)
        for i in range(new_edge_index.size(1)):
            dst = new_edge_index[1, i]
            in_degree[dst] += 1
        
        # 對於度數小於閾值的節點，添加額外的連接
        min_deg_threshold = max(self.min_deg, int(self.min_deg_ratio * torch.mean(in_degree).item()))
        
        for node in range(num_nodes):
            if in_degree[node] < min_deg_threshold:
                # 獲取該節點的注意力分數
                node_similarity = attention_scores[node].clone()  # 創建副本
                node_similarity[node] = float('-inf')
                
                # 獲取已連接的節點，避免重複
                connected_nodes = new_edge_index[1, new_edge_index[0] == node]
                for connected in connected_nodes:
                    node_similarity[connected] = float('-inf')
                
                # 計算需要添加的額外連接數量
                num_to_add = min_deg_threshold - int(in_degree[node].item())
                
                # 選擇最相似的節點作為額外連接
                _, extra_indices = torch.topk(node_similarity, num_to_add)
                
                # 創建額外的邊
                extra_src = torch.full((num_to_add,), node, device=device)
                extra_dst = extra_indices
                
                # 添加到邊索引
                additional_edges = torch.stack([extra_dst, extra_src])  # 注意這裡是反向連接，確保入度增加
                new_edge_index = torch.cat([new_edge_index, additional_edges], dim=1)
        
        # ========== 全新的同質性增強機制 ==========
        if labels is not None and self.homophily_weight > 0:
            # 獲取基準同質性
            base_homophily = homophily(new_edge_index, labels)
            if isinstance(base_homophily, torch.Tensor):
                base_homophily = base_homophily.item()
            
            # 創建快速查找數據結構
            # 1. 為每個標籤找到所有節點
            label_to_nodes = {}
            for i in range(num_nodes):
                label = int(labels[i].item())
                if label not in label_to_nodes:
                    label_to_nodes[label] = []
                label_to_nodes[label].append(i)
            
            # 2. 創建已存在邊的集合，用於快速查詢
            existing_edges = set()
            for i in range(new_edge_index.size(1)):
                src, dst = int(new_edge_index[0, i].item()), int(new_edge_index[1, i].item())
                existing_edges.add((src, dst))
            
            # 3. 計算每個節點的同質性
            node_homophily = torch.zeros(num_nodes, device=device)
            for node in range(num_nodes):
                neighbors = new_edge_index[1, new_edge_index[0] == node]
                if len(neighbors) == 0:
                    continue
                
                node_label = labels[node]
                same_label_count = 0
                for neighbor in neighbors:
                    if labels[neighbor] == node_label:
                        same_label_count += 1
                
                node_homophily[node] = same_label_count / len(neighbors)
            
            # 根據同質性權重動態設置閾值
            homophily_threshold = max(0.2, 0.4 - 0.03 * self.homophily_weight)
            
            # 找出需要增強同質性的節點
            low_homophily_nodes = torch.where(node_homophily < homophily_threshold)[0]
            
            # 控制總共添加的邊數量
            max_edges_per_node = max(2, int(1 + self.homophily_weight))
            total_max_edges = int(num_nodes * 0.1 * self.homophily_weight)
            
            # 存儲新添加的同質性邊
            homophily_edges = []
            added_edges_count = 0
            
            # 對每個低同質性節點，添加同類連接
            for node in low_homophily_nodes:
                if added_edges_count >= total_max_edges:
                    break
                
                node_idx = int(node.item())
                node_label = int(labels[node].item())
                
                # 獲取同類節點
                same_label_nodes = label_to_nodes[node_label]
                same_label_nodes = [n for n in same_label_nodes if n != node_idx]
                
                if not same_label_nodes:
                    continue
                
                # 計算與同類節點的相似度
                node_feature = q_features[node].unsqueeze(0)
                sim_scores = []
                valid_candidates = []
                
                for candidate in same_label_nodes:
                    # 排除已存在的邊
                    if (node_idx, candidate) in existing_edges:
                        continue
                    
                    # 計算相似度
                    candidate_feature = k_features[candidate].unsqueeze(0)
                    sim = torch.mm(node_feature, candidate_feature.t()).item()
                    
                    # 應用同質性權重 - 更強的權重效果
                    weighted_sim = sim * (1.0 + 0.5 * self.homophily_weight)
                    
                    sim_scores.append(weighted_sim)
                    valid_candidates.append(candidate)
                
                if not valid_candidates:
                    continue
                
                # 選擇最相似的節點建立連接
                sim_scores = torch.tensor(sim_scores, device=device)
                num_to_add = min(len(valid_candidates), max_edges_per_node)
                
                if num_to_add > 0:
                    _, top_indices = torch.topk(sim_scores, num_to_add)
                    
                    for idx in top_indices:
                        candidate = valid_candidates[idx]
                        
                        # 創建雙向連接
                        src = torch.tensor([node_idx], device=device)
                        dst = torch.tensor([candidate], device=device)
                        
                        # 添加邊
                        homophily_edges.append(torch.stack([src, dst]))
                        homophily_edges.append(torch.stack([dst, src]))  # 反向連接也添加
                        
                        # 更新已存在邊集合
                        existing_edges.add((node_idx, candidate))
                        existing_edges.add((candidate, node_idx))
                        
                        added_edges_count += 2
                        
                        if added_edges_count >= total_max_edges:
                            break
            
            # 合併同質性增強的邊
            if homophily_edges:
                homophily_edge_index = torch.cat(homophily_edges, dim=1)
                enhanced_edge_index = torch.cat([new_edge_index, homophily_edge_index], dim=1)
                
                # 計算增強後的同質性
                enhanced_homophily = homophily(enhanced_edge_index, labels)
                if isinstance(enhanced_homophily, torch.Tensor):
                    enhanced_homophily = enhanced_homophily.item()
                
                if self.training:
                    print(f"Homophily changed from {base_homophily:.4f} to {enhanced_homophily:.4f} with weight {self.homophily_weight}")
                
                # 只有在同質性真的提高時才使用增強後的圖
                if enhanced_homophily > base_homophily:
                    new_edge_index = enhanced_edge_index
                else:
                    # 如果同質性反而下降，則保持原樣
                    if self.training:
                        print(f"Homophily enhancement rejected: would decrease from {base_homophily:.4f} to {enhanced_homophily:.4f}")
        
        # 應用邊丟棄以防止過擬合
        if self.training and self.edge_dropout > 0:
            num_edges = new_edge_index.size(1)
            keep_mask = torch.rand(num_edges, device=device) > self.edge_dropout
            new_edge_index = new_edge_index[:, keep_mask]
        
        # 去除重複的邊
        new_edge_index = coalesce(new_edge_index, None, num_nodes, num_nodes)[0]
        
        return new_edge_index

    def __call__(self, data, k=None, epsilon=None, embedding_post=True, cat_self=True, 
                prunning=False, thres_prunning=0., batch_size=128, thres_min_deg=None, 
                thres_min_deg_ratio=None, drop_edge=False, prob_drop_edge=0.0, 
                window=None, shuffle=None, drop_last=None, epoch_train_gl=200, 
                epoch_finetune_gl=30, seed_gl=0):
        """
        與ModelHandler.forward方法兼容的接口
        
        Args:
            data: 圖數據對象
            k: 每個節點選擇的最大連接數
            其他參數: 與原始接口兼容，但大部分不使用
            
        Returns:
            new_data: 重連後的圖數據對象
        """
        # 更新參數（如果提供）
        if k is not None:
            self.top_k = k
        if thres_min_deg is not None:
            self.min_deg = thres_min_deg
        if thres_min_deg_ratio is not None:
            self.min_deg_ratio = thres_min_deg_ratio
        
        # 設置訓練模式
        self.training = True
        
        # 獲取節點特徵和標籤
        x = data.x
        edge_index = data.edge_index
        labels = data.y if hasattr(data, 'y') else None
        
        # 應用注意力重連
        new_edge_index = self.forward(x, edge_index, labels)
        
        # 創建新的數據對象
        new_data = copy.deepcopy(data)
        new_data.edge_index = new_edge_index
        
        return new_data