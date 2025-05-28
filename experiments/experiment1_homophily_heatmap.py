import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
from torch_geometric.utils import homophily
from torch_geometric.datasets import WikipediaNetwork
from improved_attention_rewiring import ImprovedAttentionRewirer

# 設置命令行參數
parser = argparse.ArgumentParser(description='實驗1: 同質性熱圖 (top_k vs homophily_weight)')
parser.add_argument('--dataset', type=str, default='chameleon', choices=['chameleon', 'squirrel'],
                    help='數據集名稱 (default: chameleon)')
parser.add_argument('--gpu', type=int, default=0, help='GPU 設備 ID (default: 0)')
parser.add_argument('--output_dir', type=str, default='results/exp1_homophily_heatmap',
                    help='輸出目錄 (default: results/exp1_homophily_heatmap)')
args = parser.parse_args()

# 設置設備
device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
print(f"使用設備: {device}")

# 創建輸出目錄
os.makedirs(args.output_dir, exist_ok=True)

# 載入數據集
print(f"載入 {args.dataset} 數據集...")
dataset = WikipediaNetwork(root='../data', name=args.dataset)
data = dataset[0].to(device)
print(f"數據集載入完成: {data}")

# 定義參數範圍
top_k_values = [8, 16, 32, 64]
homophily_weight_values = [0.5, 1.0, 2.0, 4.0]

# 創建結果DataFrame
results = []

# 運行實驗
print("開始實驗...")
for top_k in top_k_values:
    for homophily_weight in homophily_weight_values:
        print(f"測試 top_k={top_k}, homophily_weight={homophily_weight}")
        
        # 初始化改進的注意力重連器
        rewirer = ImprovedAttentionRewirer(
            top_k=top_k,
            temperature=0.5,  # 使用默認溫度
            min_deg=5,        # 使用默認最小度
            min_deg_ratio=1.5,  # 使用默認最小度比例
            homophily_weight=homophily_weight,
            feature_dropout=0.2,  # 使用默認特徵丟棄率
            edge_dropout=0.1      # 使用默認邊丟棄率
        )
        
        # 應用重連
        new_data = rewirer(data)
        
        # 計算同質性比率
        h_ratio = homophily(new_data.edge_index, new_data.y)
        # 檢查返回值類型，如果是張量則調用item()
        if isinstance(h_ratio, torch.Tensor):
            h_ratio = h_ratio.item()
        
        # 儲存結果
        results.append({
            'top_k': top_k,
            'homophily_weight': homophily_weight,
            'homophily_ratio': h_ratio
        })
        
        print(f"同質性比率: {h_ratio:.4f}")

# 轉換為DataFrame
df = pd.DataFrame(results)

# 保存結果到CSV
csv_path = os.path.join(args.output_dir, f"{args.dataset}_homophily_results.csv")
df.to_csv(csv_path, index=False)
print(f"結果已保存到: {csv_path}")

# 創建熱圖
print("生成熱圖...")
plt.figure(figsize=(10, 8))
pivot_table = df.pivot(index='top_k', columns='homophily_weight', values='homophily_ratio')
sns.heatmap(pivot_table, annot=True, fmt=".3f", cmap="YlGnBu", cbar_kws={'label': '同質性比率'})
plt.title(f'{args.dataset.capitalize()} 數據集: top_k vs homophily_weight 的同質性熱圖')
plt.xlabel('同質性權重 (homophily_weight)')
plt.ylabel('最大連接數 (top_k)')

# 保存熱圖
heatmap_path = os.path.join(args.output_dir, f"{args.dataset}_homophily_heatmap.png")
plt.tight_layout()
plt.savefig(heatmap_path, dpi=300)
print(f"熱圖已保存到: {heatmap_path}")

# 額外創建一個3D表面圖
print("生成3D表面圖...")
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

X, Y = np.meshgrid(homophily_weight_values, top_k_values)
Z = pivot_table.values

surf = ax.plot_surface(X, Y, Z, cmap='coolwarm', edgecolor='none', alpha=0.8)
ax.set_xlabel('同質性權重 (homophily_weight)')
ax.set_ylabel('最大連接數 (top_k)')
ax.set_zlabel('同質性比率')
ax.set_title(f'{args.dataset.capitalize()} 數據集: 參數對同質性的影響')

# 添加顏色條
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

# 保存3D圖
surface_path = os.path.join(args.output_dir, f"{args.dataset}_homophily_surface.png")
plt.tight_layout()
plt.savefig(surface_path, dpi=300)
print(f"3D表面圖已保存到: {surface_path}")

print("實驗1完成!")
