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
            node_similarity = attention_scores[node]
            
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
                node_similarity = attention_scores[node]
                node_similarity[node] = float('-inf')
                
                # 排除已經連接的節點
                connected_nodes = new_edge_index[0, new_edge_index[1] == node]
                node_similarity[connected_nodes] = float('-inf')
                
                # 獲取額外的連接數量
                additional_edges = min_deg_threshold - int(in_degree[node].item())
                
                # 獲取額外的連接
                _, extra_indices = torch.topk(node_similarity, additional_edges)
                
                # 添加額外的連接
                extra_src = torch.full((additional_edges,), node, device=device)
                extra_dst = extra_indices
                
                # 添加到邊索引
                additional_edges = torch.stack([extra_dst, extra_src])  # 注意這裡是反向連接，確保入度增加
                new_edge_index = torch.cat([new_edge_index, additional_edges], dim=1)
        
        # 增強同質性：如果有標籤信息，優先連接同類節點
        if labels is not None:
            # 計算當前同質性
            current_homophily = homophily(new_edge_index, labels)
            
            # 如果同質性較低，增加同類節點之間的連接
            if current_homophily < 0.5:  # 可調整的閾值
                # 為每個節點找到同類節點
                for node in range(num_nodes):
                    node_label = labels[node]
                    same_label_nodes = torch.where(labels == node_label)[0]
                    
                    # 排除自身
                    same_label_nodes = same_label_nodes[same_label_nodes != node]
                    
                    if len(same_label_nodes) > 0:
                        # 計算與同類節點的相似度
                        similarity = torch.mm(q_features[node].unsqueeze(0), 
                                             k_features[same_label_nodes].t()).squeeze()
                        
                        # 選擇最相似的同類節點
                        num_to_add = min(3, len(same_label_nodes))  # 每個節點最多添加3個同類連接
                        _, top_indices = torch.topk(similarity, num_to_add)
                        top_same_label_nodes = same_label_nodes[top_indices]
                        
                        # 添加同類連接
                        src = torch.full((num_to_add,), node, device=device)
                        dst = top_same_label_nodes
                        
                        # 添加到邊索引
                        same_label_edges = torch.stack([src, dst])
                        new_edge_index = torch.cat([new_edge_index, same_label_edges], dim=1)
        
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
