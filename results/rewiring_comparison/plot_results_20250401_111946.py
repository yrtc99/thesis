import json
import matplotlib.pyplot as plt
import numpy as np
import os

def load_results(results_file):
    """載入實驗結果"""
    with open(results_file, 'r') as f:
        return json.load(f)

def plot_comparison(results_data, metric='test', show_std=True):
    """繪製不同重連方法的比較圖"""
    dataset = results_data['dataset']
    model = results_data['model']
    results = results_data['results']
    
    # 準備數據
    methods = list(results.keys())
    avg_values = [results[method]['avg'][metric] for method in methods]
    std_values = [results[method]['std'][metric] for method in methods] if show_std else None
    
    # 創建柱狀圖
    plt.figure(figsize=(10, 6))
    bars = plt.bar(methods, avg_values, yerr=std_values, capsize=5, alpha=0.7)
    
    # 添加數值標籤
    for bar, val in zip(bars, avg_values):
        plt.text(bar.get_x() + bar.get_width()/2, val + 0.01, f'{val:.4f}', 
                 ha='center', va='bottom', fontsize=10)
    
    # 設置標題和標籤
    plt.title(f'{dataset} - {model} - {metric.capitalize()} Accuracy Comparison')
    plt.ylabel(f'{metric.capitalize()} Accuracy')
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    return plt

def plot_all_metrics(results_data, show_std=True):
    """繪製所有指標的比較圖"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    metrics = ['train', 'val', 'test']
    
    for i, metric in enumerate(metrics):
        dataset = results_data['dataset']
        model = results_data['model']
        results = results_data['results']
        
        # 準備數據
        methods = list(results.keys())
        avg_values = [results[method]['avg'][metric] for method in methods]
        std_values = [results[method]['std'][metric] for method in methods] if show_std else None
        
        # 創建柱狀圖
        bars = axes[i].bar(methods, avg_values, yerr=std_values, capsize=5, alpha=0.7)
        
        # 添加數值標籤
        for bar, val in zip(bars, avg_values):
            axes[i].text(bar.get_x() + bar.get_width()/2, val + 0.01, f'{val:.4f}', 
                     ha='center', va='bottom', fontsize=10)
        
        # 設置標題和標籤
        axes[i].set_title(f'{metric.capitalize()} Accuracy')
        axes[i].set_ylim(0, 1.0)
        axes[i].grid(axis='y', linestyle='--', alpha=0.7)
    
    fig.suptitle(f'{dataset} - {model} - Performance Comparison', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    return fig

# 示例使用
def find_latest_result(results_dir='./results/rewiring_comparison'):
    """找到最新的結果文件"""
    json_files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
    if not json_files:
        return None
    
    latest_file = max(json_files, key=lambda x: os.path.getmtime(os.path.join(results_dir, x)))
    return os.path.join(results_dir, latest_file)

# 使用示例
'''
latest_result = find_latest_result()
if latest_result:
    results_data = load_results(latest_result)
    
    # 繪製測試準確率比較圖
    plt.figure(1)
    plot_comparison(results_data, metric='test')
    
    # 繪製所有指標比較圖
    plt.figure(2)
    plot_all_metrics(results_data)
    
    plt.show()
else:
    print("未找到結果文件")
'''
