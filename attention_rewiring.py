import torch
import torch.nn as nn
import torch_sparse
from torch_sparse import SparseTensor, set_diag
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch_geometric.utils import coalesce
import copy


class AttentionBasedRewiring(nn.Module):
    """基於注意力機制的圖重連模塊，與DHGR框架整合"""
    
    def __init__(self, in_size, hidden_dim=64, temperature=1.0, top_k=8, 
                 num_layer=2, dropout=0.0, use_center_moment=False, moment=1, device=None):
        """
        初始化注意力重連模塊
        
        參數:
            in_size: 輸入特徵維度
            hidden_dim: 隱藏層維度
            temperature: 注意力分數溫度參數
            top_k: 每個節點選擇的最大連接數
            num_layer: 注意力層數
            dropout: dropout比率
            use_center_moment: 是否使用中心矩
            moment: 矩的階數
            device: 計算設備
        """
        super(AttentionBasedRewiring, self).__init__()
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        self.top_k = top_k
        self.num_layer = num_layer
        self.dropout = dropout
        self.use_center_moment = use_center_moment
        self.moment = moment
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 特徵投影層
        self.node_projection_q = nn.Linear(in_size, hidden_dim)
        self.node_projection_k = nn.Linear(in_size, hidden_dim)
        
        # 注意力打分層 (多層)
        self.attention_layers = nn.ModuleList()
        for i in range(num_layer):
            if i == 0:
                self.attention_layers.append(nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.LeakyReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim // 2)
                ))
            elif i == num_layer - 1:
                self.attention_layers.append(nn.Sequential(
                    nn.Linear(hidden_dim // 2, hidden_dim // 4),
                    nn.LeakyReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim // 4, 1)
                ))
            else:
                self.attention_layers.append(nn.Sequential(
                    nn.Linear(hidden_dim // 2, hidden_dim // 2),
                    nn.LeakyReLU(),
                    nn.Dropout(dropout)
                ))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """重置模型參數"""
        nn.init.xavier_uniform_(self.node_projection_q.weight)
        nn.init.xavier_uniform_(self.node_projection_k.weight)
        for layer in self.attention_layers:
            for module in layer:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
    
    def compute_moment_features(self, x):
        """計算特徵的矩"""
        if not self.use_center_moment or self.moment <= 1:
            return x
        
        # 計算中心矩
        mean = torch.mean(x, dim=0, keepdim=True)
        centered_x = x - mean
        
        # 計算指定階數的矩
        moment_features = torch.pow(centered_x, self.moment)
        
        # 合併原始特徵和矩特徵
        return torch.cat([x, moment_features], dim=1)
    
    def forward(self, data, embedding=True, cat_self=True, batch_size=128, thres_min_deg=3, thres_min_deg_ratio=1.0, drop_edge=False, prob_drop_edge=0.0):
        """
        前向傳播: 計算注意力重連
        
        參數:
            data: 圖數據對象
            embedding: 是否使用嵌入特徵
            cat_self: 是否包含自環
            batch_size: 批處理大小
            thres_min_deg: 最小度閾值
            thres_min_deg_ratio: 最小度比例閾值
            drop_edge: 是否隨機丟棄邊
            prob_drop_edge: 丟棄邊的概率
            
        回傳:
            new_edge_index: 重連後的邊索引
        """
        x = data.x.to(self.device)
        
        # 應用矩特徵（如果啟用）
        if self.use_center_moment and self.moment > 1:
            x = self.compute_moment_features(x)
        
        # 節點特徵投影
        q_features = self.node_projection_q(x)
        k_features = self.node_projection_k(x)
        
        # 計算注意力分數矩陣 (批處理方式)
        num_nodes = x.shape[0]
        
        all_src_indices = []
        all_dst_indices = []
        all_attention_scores = []
        
        for i in range(0, num_nodes, batch_size):
            # 取當前批次的節點
            batch_q = q_features[i:i+batch_size]
            batch_size_actual = batch_q.shape[0]
            
            # 計算當前批次與所有節點的注意力分數
            similarity = torch.mm(batch_q, k_features.t()) / self.temperature
            
            # 獲取每個節點連接到的前K個目標節點
            if self.top_k < num_nodes:
                topk_values, topk_indices = torch.topk(similarity, min(self.top_k, num_nodes), dim=1)
                
                # 創建源節點索引
                batch_src_indices = torch.arange(i, i + batch_size_actual, device=self.device)
                batch_src_indices = batch_src_indices.repeat_interleave(topk_indices.shape[1])
                
                # 創建目標節點索引
                batch_dst_indices = topk_indices.reshape(-1)
                
                # 收集注意力分數
                batch_attention_scores = topk_values.reshape(-1)
                
                all_src_indices.append(batch_src_indices)
                all_dst_indices.append(batch_dst_indices)
                all_attention_scores.append(batch_attention_scores)
            else:
                # 如果top_k大於等於節點數，則連接所有節點
                batch_src_indices = torch.arange(i, i + batch_size_actual, device=self.device)
                batch_src_indices = batch_src_indices.repeat_interleave(num_nodes)
                
                batch_dst_indices = torch.arange(num_nodes, device=self.device).repeat(batch_size_actual)
                
                batch_attention_scores = similarity.reshape(-1)
                
                all_src_indices.append(batch_src_indices)
                all_dst_indices.append(batch_dst_indices)
                all_attention_scores.append(batch_attention_scores)
        
        # 合併所有批次的結果
        src_indices = torch.cat(all_src_indices)
        dst_indices = torch.cat(all_dst_indices)
        attention_scores = torch.cat(all_attention_scores)
        
        # 創建新的邊索引
        new_edge_index = torch.stack([src_indices, dst_indices])
        
        # 應用最小度閾值
        if thres_min_deg > 0 or thres_min_deg_ratio > 0:
            # 計算每個節點的度
            node_degrees = torch.bincount(new_edge_index[0], minlength=num_nodes)
            
            # 計算度閾值
            min_deg_threshold = max(thres_min_deg, int(thres_min_deg_ratio * torch.mean(node_degrees.float()).item()))
            
            # 找出度數低於閾值的節點
            low_degree_nodes = torch.where(node_degrees < min_deg_threshold)[0]
            
            # 為低度節點添加更多連接
            if low_degree_nodes.numel() > 0:
                for node in low_degree_nodes:
                    # 計算與其他節點的相似度
                    node_q = q_features[node:node+1]
                    node_similarity = torch.mm(node_q, k_features.t()) / self.temperature
                    
                    # 獲取額外的連接
                    _, extra_indices = torch.topk(node_similarity, min_deg_threshold, dim=1)
                    
                    # 添加額外的連接
                    extra_src = node.repeat(min_deg_threshold)
                    extra_dst = extra_indices.reshape(-1)
                    
                    new_edge_index = torch.cat([
                        new_edge_index,
                        torch.stack([extra_src, extra_dst])
                    ], dim=1)
        
        # 限制邊的數量，防止過度連接並提高同質性
        if new_edge_index.size(1) > num_nodes * self.top_k:  # 減少邊的數量，每個節點平均有 top_k 條邊
            # 計算每條邊的注意力分數
            src_nodes = new_edge_index[0]
            dst_nodes = new_edge_index[1]
            edge_scores = torch.sum(q_features[src_nodes] * k_features[dst_nodes], dim=1) / self.temperature
            
            # 增加同質性權重：如果連接的節點標籤相同，則增加其分數
            if labels is not None:
                src_labels = labels[src_nodes]
                dst_labels = labels[dst_nodes]
                same_label_mask = (src_labels == dst_labels).float()
                edge_scores = edge_scores + same_label_mask * 2.0  # 增加同標籤連接的權重
            
            # 保留分數最高的邊
            max_edges = num_nodes * self.top_k
            if edge_scores.size(0) > max_edges:
                _, top_indices = torch.topk(edge_scores, max_edges)
                new_edge_index = new_edge_index[:, top_indices]
        
        # 隨機丟棄邊（如果啟用）
        if drop_edge and prob_drop_edge > 0:
            mask = torch.rand(new_edge_index.size(1), device=self.device) >= prob_drop_edge
            new_edge_index = new_edge_index[:, mask]
        
        # 移除自環並重新添加（如果需要）
        if cat_self:
            new_edge_index, _ = remove_self_loops(new_edge_index)
            new_edge_index, _ = add_self_loops(new_edge_index, num_nodes=num_nodes)
        
        # 合併重複的邊
        new_edge_index = coalesce(new_edge_index, None, num_nodes, num_nodes)[0]
        
        # 創建新的數據對象
        new_data = copy.deepcopy(data)
        new_data.edge_index = new_edge_index
        
        return new_data


# 與DHGR整合的接口
class AttentionRewirer:
    """與DHGR框架整合的注意力重連器"""
    
    def __init__(self, in_size, hidden_dim=64, top_k=8, temperature=1.0, 
                 num_layer=2, dropout=0.0, use_center_moment=False, moment=1, device=None):
        super(AttentionRewirer, self).__init__()
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.rewiring = AttentionBasedRewiring(
            in_size=in_size,
            hidden_dim=hidden_dim,
            temperature=temperature,
            top_k=top_k,
            num_layer=num_layer,
            dropout=dropout,
            use_center_moment=use_center_moment,
            moment=moment,
            device=self.device
        )
    
    def __call__(self, data, k=8, epsilon=None, embedding_post=True, cat_self=True, 
                prunning=False, thres_prunning=0., batch_size=128, thres_min_deg=3, 
                thres_min_deg_ratio=1.0, drop_edge=False, prob_drop_edge=0.0, 
                window=None, shuffle=None, drop_last=None, epoch_train_gl=200, 
                epoch_finetune_gl=30, seed_gl=0):
        """
        與ModelHandler.forward方法兼容的接口
        
        參數:
            data: 圖數據對象
            k: 每個節點選擇的最大連接數
            epsilon: 不使用
            embedding_post: 是否使用嵌入特徵
            cat_self: 是否包含自環
            prunning: 是否修剪原始圖
            thres_prunning: 修剪閾值
            batch_size: 批處理大小
            thres_min_deg: 最小度閾值
            thres_min_deg_ratio: 最小度比例閾值
            drop_edge: 是否隨機丟棄邊
            prob_drop_edge: 丟棄邊的概率
            window: 窗口大小（不使用）
            shuffle: 是否打亂（不使用）
            drop_last: 是否丟棄最後一個批次（不使用）
            epoch_train_gl: 圖學習訓練輪數（不使用）
            epoch_finetune_gl: 圖學習微調輪數（不使用）
            seed_gl: 圖學習隨機種子（不使用）
            
        回傳:
            new_data: 重連後的圖數據對象
        """
        # 更新rewiring的top_k參數
        self.rewiring.top_k = k
        
        # 應用注意力重連
        new_data = self.rewiring(data, embedding=embedding_post, cat_self=cat_self, 
                                batch_size=batch_size, thres_min_deg=thres_min_deg,
                                thres_min_deg_ratio=thres_min_deg_ratio,
                                drop_edge=drop_edge, prob_drop_edge=prob_drop_edge)
        
        # 如果需要修剪原始圖
        if prunning and thres_prunning > 0:
            # 獲取原始邊索引
            orig_edge_index = data.edge_index
            
            # 計算原始邊的注意力分數
            x = data.x.to(self.device)
            q_features = self.rewiring.node_projection_q(x)
            k_features = self.rewiring.node_projection_k(x)
            
            # 計算每條邊的注意力分數
            src_nodes = orig_edge_index[0]
            dst_nodes = orig_edge_index[1]
            
            src_features = q_features[src_nodes]
            dst_features = k_features[dst_nodes]
            
            # 計算注意力分數
            edge_scores = torch.sum(src_features * dst_features, dim=1) / self.rewiring.temperature
            edge_scores = torch.sigmoid(edge_scores)
            
            # 根據閾值保留邊
            mask = edge_scores >= thres_prunning
            pruned_edge_index = orig_edge_index[:, mask]
            
            # 合併修剪後的原始邊和新生成的邊
            combined_edge_index = torch.cat([pruned_edge_index, new_data.edge_index], dim=1)
            combined_edge_index = coalesce(combined_edge_index, None, data.num_nodes, data.num_nodes)[0]
            
            new_data.edge_index = combined_edge_index
        
        return new_data