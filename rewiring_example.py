import torch
import torch.nn.functional as F
import argparse
from torch_geometric.utils import coalesce
from torch_geometric.datasets import Planetoid, WebKB, WikipediaNetwork, Actor
from torch_geometric.transforms import NormalizeFeatures
from models.model import GCNNet, GraphSAGE, GAT
from attention_rewiring import AttentionRewirer
from spectral_rewiring import SpectralRewirer
import numpy as np
import random
import time


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)


def train(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    # 檢查 train_mask 的形狀，確保它是一維的
    if len(data.train_mask.shape) > 1:
        # 如果是多維的，取第一個分割（split）
        train_mask = data.train_mask[:, 0]
    else:
        train_mask = data.train_mask
    
    loss = F.nll_loss(out[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def test(model, data):
    model.eval()
    pred = model(data).argmax(dim=1)
    
    accs = []
    for i, mask_name in enumerate(['train_mask', 'val_mask', 'test_mask']):
        mask = getattr(data, mask_name)
        # 檢查 mask 的形狀，確保它是一維的
        if len(mask.shape) > 1:
            # 如果是多維的，取第一個分割（split）
            mask = mask[:, 0]
        
        correct = (pred[mask] == data.y[mask]).sum().item()
        total = mask.sum().item()
        acc = correct / total
        accs.append(acc)
    
    return accs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cora', 
                        choices=['Cora', 'CiteSeer', 'PubMed', 'Cornell', 'Texas', 'Wisconsin', 'Chameleon', 'Squirrel', 'Actor'])
    parser.add_argument('--model', type=str, default='GCN', choices=['GCN', 'SAGE', 'GAT'])
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--rewiring', type=str, default='none', 
                        choices=['none', 'attention', 'spectral', 'both'])
    parser.add_argument('--top_k', type=int, default=8, 
                        help='Number of neighbors for rewiring')
    parser.add_argument('--prune', action='store_true', 
                        help='Whether to prune original edges')
    parser.add_argument('--prune_threshold', type=float, default=0.0, 
                        help='Threshold for pruning')
    args = parser.parse_args()
    
    # 設置隨機種子
    set_random_seed(args.seed)
    
    # 設置設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加載數據集
    if args.dataset in ['Cora', 'CiteSeer', 'PubMed']:
        dataset = Planetoid(root='/tmp/' + args.dataset, name=args.dataset, transform=NormalizeFeatures())
    elif args.dataset in ['Cornell', 'Texas', 'Wisconsin']:
        dataset = WebKB(root='/tmp/' + args.dataset, name=args.dataset, transform=NormalizeFeatures())
    elif args.dataset in ['Chameleon', 'Squirrel']:
        dataset = WikipediaNetwork(root='/tmp/' + args.dataset, name=args.dataset, transform=NormalizeFeatures())
    elif args.dataset == 'Actor':
        dataset = Actor(root='/tmp/Actor', transform=NormalizeFeatures())
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    data = dataset[0].to(device)
    
    # 應用圖重連
    if args.rewiring != 'none':
        print(f"Applying {args.rewiring} rewiring...")
        
        if args.rewiring == 'attention' or args.rewiring == 'both':
            attention_rewirer = AttentionRewirer(
                in_size=dataset.num_features,
                hidden_dim=args.hidden_channels,
                top_k=args.top_k,
                device=device
            )
            data_attention = attention_rewirer(
                data,
                k=args.top_k,
                prunning=args.prune,
                thres_prunning=args.prune_threshold
            )
            
            if args.rewiring == 'attention':
                data = data_attention
        
        if args.rewiring == 'spectral' or args.rewiring == 'both':
            spectral_rewirer = SpectralRewirer(
                in_size=dataset.num_features,
                hidden_dim=args.hidden_channels,
                k_neighbors=args.top_k,
                device=device
            )
            data_spectral = spectral_rewirer(
                data,
                k=args.top_k,
                prunning=args.prune,
                thres_prunning=args.prune_threshold
            )
            
            if args.rewiring == 'spectral':
                data = data_spectral
        
        if args.rewiring == 'both':
            # 合併兩種重連的結果
            edge_index_combined = torch.cat([data_attention.edge_index, data_spectral.edge_index], dim=1)
            edge_index_combined = coalesce(edge_index_combined, None, data.num_nodes, data.num_nodes)[0]
            data.edge_index = edge_index_combined
        
        print(f"Original edges: {dataset[0].edge_index.size(1)}, Rewired edges: {data.edge_index.size(1)}")
    
    # 創建模型
    if args.model == 'GCN':
        model = GCNNet(dataset, layer_num=2, hidden=args.hidden_channels).to(device)
    elif args.model == 'SAGE':
        model = GraphSAGE(dataset, layer_num=2, hidden=args.hidden_channels).to(device)
    elif args.model == 'GAT':
        model = GAT(dataset, hidden=args.hidden_channels).to(device)
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    # 優化器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    
    # 訓練和評估
    best_val_acc = 0
    best_test_acc = 0
    for epoch in range(1, args.epochs + 1):
        loss = train(model, data, optimizer)
        train_acc, val_acc, test_acc = test(model, data)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
        
        if epoch % 20 == 0:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')
    
    print(f'Best validation accuracy: {best_val_acc:.4f}')
    print(f'Best test accuracy: {best_test_acc:.4f}')


if __name__ == '__main__':
    main()
