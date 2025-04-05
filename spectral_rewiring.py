import torch
import torch.nn as nn
import torch_sparse
from torch_sparse import SparseTensor, coalesce
from torch_geometric.utils import remove_self_loops, add_self_loops
import copy
import numpy as np


class SpectralRewiring(nn.Module):
    """基於譜分析的圖重連模塊，與DHGR框架整合"""
    
    def __init__(self, in_size, hidden_dim=64, k_neighbors=8, 
                 num_layer=2, dropout=0.0, use_center_moment=False, moment=1, device=None):
        """
        初始化譜重連模塊
        
        參數:
            in_size: 輸入特徵維度
            hidden_dim: 隱藏層維度
            k_neighbors: 每個節點選擇的最近鄰數量
            num_layer: 隱藏層數量
            dropout: dropout率
            use_center_moment: 是否使用中心化矩
            moment: 矩的階數
            device: 計算設備
        """
        super(SpectralRewiring, self).__init__()
        self.hidden_dim = hidden_dim
        self.k_neighbors = k_neighbors
        self.num_layer = num_layer
        self.dropout = dropout
        self.use_center_moment = use_center_moment
        self.moment = moment
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 創建譜嵌入層
        self.spectral_embedding = nn.ModuleList()
        for i in range(num_layer):
            if i == 0:
                self.spectral_embedding.append(nn.Sequential(
                    nn.Linear(in_size, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim)
                ))
            else:
                self.spectral_embedding.append(nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim)
                ))
        
        # 創建Graph Fourier Transform (GFT)
        self.gft_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """重置模型參數"""
        for layer in self.spectral_embedding:
            for module in layer:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
        
        for layer in self.gft_layer:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
    
    def compute_laplacian(self, edge_index, num_nodes):
        """
        計算圖的拉普拉斯矩陣
        
        參數:
            edge_index: 邊索引
            num_nodes: 節點數量
            
        回傳:
            L: 拉普拉斯矩陣
        """
        # 創建鄰接矩陣
        adj = SparseTensor(row=edge_index[0], col=edge_index[1], 
                        value=torch.ones(edge_index.shape[1], device=self.device),
                        sparse_sizes=(num_nodes, num_nodes))
        
        # 計算度矩陣
        deg = torch_sparse.sum(adj, dim=1)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        
        # 計算正規化拉普拉斯矩陣 L = I - D^(-1/2) A D^(-1/2)
        adj_norm = torch_sparse.mul(adj, deg_inv_sqrt.view(-1, 1))
        adj_norm = torch_sparse.mul(adj_norm, deg_inv_sqrt.view(1, -1))
        
        # L = I - D^(-1/2) A D^(-1/2)
        identity = SparseTensor.eye(num_nodes, device=self.device)
        lap = identity - adj_norm
        
        return lap
    
    def compute_moment_features(self, x):
        """計算矩特徵"""
        if not self.use_center_moment or self.moment <= 1:
            return x
        
        # 計算中心化矩
        mean = torch.mean(x, dim=0, keepdim=True)
        centered_x = x - mean
        
        # 計算階數矩
        moment_features = torch.pow(centered_x, self.moment)
        
        # 結合原始特徵和矩特徵
        return torch.cat([x, moment_features], dim=1)
    
    def forward(self, data, embedding=True, cat_self=True, batch_size=128, thres_min_deg=3, 
                thres_min_deg_ratio=1.0, drop_edge=False, prob_drop_edge=0.0):
        """
        前向傳播: 計算譜重連
        
        參數:
            data: 圖數據對象
            embedding: 是否使用嵌入
            cat_self: 是否結合自身邊
            batch_size: 批次大小
            thres_min_deg: 最小度閾值
            thres_min_deg_ratio: 最小度比率閾值
            drop_edge: 是否丟棄邊
            prob_drop_edge: 丟棄邊的概率
            
        回傳:
            new_data: 重連後的圖數據對象
        """
        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)
        num_nodes = x.shape[0]
        
        # 使用矩特徵
        if self.use_center_moment and self.moment > 1:
            x = self.compute_moment_features(x)
        
        # 計算拉普拉斯矩陣
        lap = self.compute_laplacian(edge_index, num_nodes)
        
        # 使用嵌入
        h = x
        for i, layer in enumerate(self.spectral_embedding):
            h = layer(h)
        
        # 使用GFT層
        gft_x = self.gft_layer(h)
        
        # 使用譜濾波
        filtered_x = torch_sparse.matmul(lap, gft_x)
        
        # 計算相似度矩陣
        similarity_matrix = torch.mm(filtered_x, filtered_x.t())
        
        # 批次處理
        all_src_indices = []
        all_dst_indices = []
        
        for i in range(0, num_nodes, batch_size):
            # 取當前批次的源節點
            batch_size_actual = min(batch_size, num_nodes - i)
            batch_indices = torch.arange(i, i + batch_size_actual, device=self.device)
            
            # 計算當前批次與所有目標節點的相似度
            batch_similarity = similarity_matrix[batch_indices]
            
            # 選擇每個節點的k個最近鄰居
            if self.k_neighbors < num_nodes:
                _, topk_indices = torch.topk(batch_similarity, min(self.k_neighbors, num_nodes), dim=1)
                
                # 創建源節點索引
                batch_src_indices = batch_indices.repeat_interleave(topk_indices.shape[1])
                
                # 創建目標節點索引
                batch_dst_indices = topk_indices.reshape(-1)
                
                all_src_indices.append(batch_src_indices)
                all_dst_indices.append(batch_dst_indices)
            else:
                # 如果k_neighbors大於節點數量，則選擇所有節點
                batch_src_indices = batch_indices.repeat_interleave(num_nodes)
                batch_dst_indices = torch.arange(num_nodes, device=self.device).repeat(batch_size_actual)
                
                all_src_indices.append(batch_src_indices)
                all_dst_indices.append(batch_dst_indices)
        
        # 合併所有批次的結果
        src_indices = torch.cat(all_src_indices)
        dst_indices = torch.cat(all_dst_indices)
        
        # 創建新的邊索引
        new_edge_index = torch.stack([src_indices, dst_indices])
        
        # 應用最小度閾值
        if thres_min_deg > 0 or thres_min_deg_ratio > 0:
            # 計算節點度
            node_degrees = torch.bincount(new_edge_index[0], minlength=num_nodes)
            
            # 計算最小度閾值
            min_deg_threshold = max(thres_min_deg, int(thres_min_deg_ratio * torch.mean(node_degrees).item()))
            
            # 選擇度大於閾值的節點
            low_degree_nodes = torch.where(node_degrees < min_deg_threshold)[0]
            
            # 為低度節點添加額外邊
            if low_degree_nodes.numel() > 0:
                for node in low_degree_nodes:
                    # 計算節點與所有其他節點的相似度
                    node_similarity = similarity_matrix[node]
                    
                    # 選擇最相似的節點
                    _, extra_indices = torch.topk(node_similarity, min_deg_threshold, dim=0)
                    
                    # 創建額外邊
                    extra_src = node.repeat(min_deg_threshold)
                    extra_dst = extra_indices.reshape(-1)
                    
                    new_edge_index = torch.cat([
                        new_edge_index,
                        torch.stack([extra_src, extra_dst])
                    ], dim=1)
        
        # 應用丟棄邊
        if drop_edge and prob_drop_edge > 0:
            mask = torch.rand(new_edge_index.size(1), device=self.device) >= prob_drop_edge
            new_edge_index = new_edge_index[:, mask]
        
        # 結合自身邊
        if cat_self:
            new_edge_index, _ = remove_self_loops(new_edge_index)
            new_edge_index, _ = add_self_loops(new_edge_index, num_nodes=num_nodes)
        
        # 合併邊
        new_edge_index = coalesce(new_edge_index, None, num_nodes, num_nodes)[0]
        
        # 創建新的圖數據對象
        new_data = copy.deepcopy(data)
        new_data.edge_index = new_edge_index
        
        return new_data


# 與DHGR整合的接口
class SpectralRewirer(nn.Module):
    """與DHGR框架整合的譜重連器"""
    
    def __init__(self, in_size, hidden_dim=64, k_neighbors=8, 
                 num_layer=2, dropout=0.0, use_center_moment=False, moment=1, device=None):
        super(SpectralRewirer, self).__init__()
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.rewiring = SpectralRewiring(
            in_size=in_size,
            hidden_dim=hidden_dim,
            k_neighbors=k_neighbors,
            num_layer=num_layer,
            dropout=dropout,
            use_center_moment=use_center_moment,
            moment=moment,
            device=self.device
        )
    
    def forward(self, data, k=8, epsilon=None, embedding_post=True, cat_self=True, 
                prunning=False, thres_prunning=0., batch_size=128, thres_min_deg=3, 
                thres_min_deg_ratio=1.0, drop_edge=False, prob_drop_edge=0.0, 
                window=None, shuffle=None, drop_last=None, epoch_train_gl=200, 
                epoch_finetune_gl=30, seed_gl=0):
        """
        與ModelHandler.forward方法兼容的接口
        
        參數:
            data: 圖數據對象
            k: 每個節點選擇的最近鄰數量
            epsilon: 不使用
            embedding_post: 是否使用嵌入
            cat_self: 是否結合自身邊
            prunning: 是否修剪邊
            thres_prunning: 修剪閾值
            batch_size: 批次大小
            thres_min_deg: 最小度閾值
            thres_min_deg_ratio: 最小度比率閾值
            drop_edge: 是否丟棄邊
            prob_drop_edge: 丟棄邊的概率
            window: 不使用
            shuffle: 不使用
            drop_last: 不使用
            epoch_train_gl: 不使用
            epoch_finetune_gl: 不使用
            seed_gl: 不使用
            
        回傳:
            new_data: 重連後的圖數據對象
        """
        # 更新rewiring的k_neighbors參數
        self.rewiring.k_neighbors = k
        
        # 使用譜重連
        new_data = self.rewiring(data, embedding=embedding_post, cat_self=cat_self, 
                                batch_size=batch_size, thres_min_deg=thres_min_deg,
                                thres_min_deg_ratio=thres_min_deg_ratio,
                                drop_edge=drop_edge, prob_drop_edge=prob_drop_edge)
        
        # 應用修剪邊
        if prunning and thres_prunning > 0:
            # 取原始邊索引
            orig_edge_index = data.edge_index
            
            # 計算原始邊的特徵
            x = data.x.to(self.device)
            
            # 計算拉普拉斯矩陣
            lap = self.rewiring.compute_laplacian(orig_edge_index, x.shape[0])
            
            # 計算嵌入
            h = x
            for layer in self.rewiring.spectral_embedding:
                h = layer(h)
            
            # 計算原始邊的特徵
            src_nodes = orig_edge_index[0]
            dst_nodes = orig_edge_index[1]
            
            src_features = h[src_nodes]
            dst_features = h[dst_nodes]
            
            # 計算邊的相似度
            edge_scores = torch.sum(src_features * dst_features, dim=1)
            edge_scores = torch.sigmoid(edge_scores)
            
            # 選擇相似度大於閾值的邊
            mask = edge_scores >= thres_prunning
            pruned_edge_index = orig_edge_index[:, mask]
            
            # 合併修剪後的原始邊和譜重連的邊
            combined_edge_index = torch.cat([pruned_edge_index, new_data.edge_index], dim=1)
            combined_edge_index = coalesce(combined_edge_index, None, data.num_nodes, data.num_nodes)[0]
            
            new_data.edge_index = combined_edge_index
        
        return new_data