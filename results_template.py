#!/usr/bin/env python3
"""
Template for Jupyter notebook to analyze rewiring results.
Copy and paste this into your results.ipynb notebook.
"""

# Import necessary libraries
import os
import re
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define datasets and models
DATASETS = ["chameleon", "squirrel", "actor", "flickr", "fb100", "cornell", "texas", "wisconsin"]
MODELS = ["GCN", "GAT", "GraphSAGE", "APPNP", "GCNII", "H2GCN"]

def extract_metrics_from_file(file_path):
    """Extract metrics from experiment output file."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            
            # Extract accuracy
            accuracy_match = re.search(r'\[Best Score\] \{\'train\': [\d\.]+, \'val\': [\d\.]+, \'test\': ([\d\.]+)', content)
            accuracy = float(accuracy_match.group(1)) if accuracy_match else None
            
            # Extract homophily ratio
            homophily_match = re.search(r'Homophily ratio change from ([\d\.]+) to ([\d\.]+)', content)
            if homophily_match:
                original_homophily = float(homophily_match.group(1))
                new_homophily = float(homophily_match.group(2))
            else:
                # If no rewiring was done, try to find the original homophily
                homophily_match = re.search(r'Homophily ratio: ([\d\.]+)', content)
                original_homophily = float(homophily_match.group(1)) if homophily_match else None
                new_homophily = original_homophily
            
            # Extract edge count
            edge_count_match = re.search(r'Edge count: (\d+)', content)
            edge_count = int(edge_count_match.group(1)) if edge_count_match else None
            
            # Original edge count might be in a different format
            original_edge_match = re.search(r'\[Old Data\].*edge_index=\[2, (\d+)\]', content)
            original_edge_count = int(original_edge_match.group(1)) if original_edge_match else edge_count
            
            return {
                'accuracy': accuracy,
                'original_homophily': original_homophily,
                'new_homophily': new_homophily,
                'original_edge_count': original_edge_count,
                'new_edge_count': edge_count
            }
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return {
            'accuracy': None,
            'original_homophily': None,
            'new_homophily': None,
            'original_edge_count': None,
            'new_edge_count': None
        }

def create_comparison_dataframes():
    """Create DataFrames comparing original and rewired results."""
    # Initialize result dictionaries
    accuracy_original = {dataset: {} for dataset in DATASETS}
    accuracy_rewired = {dataset: {} for dataset in DATASETS}
    homophily_original = {dataset: {} for dataset in DATASETS}
    homophily_rewired = {dataset: {} for dataset in DATASETS}
    edge_count_original = {dataset: {} for dataset in DATASETS}
    edge_count_rewired = {dataset: {} for dataset in DATASETS}
    
    # Process all result files
    results_dir = "results/comparison"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"Created directory {results_dir}. Please run experiments first.")
        return {}
    
    for dataset in DATASETS:
        for model in MODELS:
            # Original results
            original_file = f"{results_dir}/{dataset}_{model}_original.txt"
            if os.path.exists(original_file):
                metrics = extract_metrics_from_file(original_file)
                accuracy_original[dataset][model] = metrics['accuracy']
                homophily_original[dataset][model] = metrics['original_homophily']
                edge_count_original[dataset][model] = metrics['original_edge_count']
            
            # Rewired results
            rewired_file = f"{results_dir}/{dataset}_{model}_attention_improved.txt"
            if os.path.exists(rewired_file):
                metrics = extract_metrics_from_file(rewired_file)
                accuracy_rewired[dataset][model] = metrics['accuracy']
                homophily_rewired[dataset][model] = metrics['new_homophily']
                edge_count_rewired[dataset][model] = metrics['new_edge_count']
    
    # Create DataFrames
    df_accuracy_original = pd.DataFrame(accuracy_original)
    df_accuracy_rewired = pd.DataFrame(accuracy_rewired)
    df_homophily_original = pd.DataFrame(homophily_original)
    df_homophily_rewired = pd.DataFrame(homophily_rewired)
    df_edge_count_original = pd.DataFrame(edge_count_original)
    df_edge_count_rewired = pd.DataFrame(edge_count_rewired)
    
    # Calculate accuracy gain
    df_accuracy_gain = df_accuracy_rewired - df_accuracy_original
    
    # Calculate homophily gain
    df_homophily_gain = df_homophily_rewired - df_homophily_original
    
    # Calculate edge count ratio
    df_edge_ratio = df_edge_count_rewired / df_edge_count_original
    
    # Calculate average gain per model
    avg_accuracy_gain = df_accuracy_gain.mean(axis=1)
    avg_homophily_gain = df_homophily_gain.mean(axis=1)
    
    # Create a summary DataFrame
    summary_data = {
        'Avg Accuracy Gain': avg_accuracy_gain,
        'Avg Homophily Gain': avg_homophily_gain
    }
    df_summary = pd.DataFrame(summary_data)
    
    return {
        'accuracy_original': df_accuracy_original,
        'accuracy_rewired': df_accuracy_rewired,
        'accuracy_gain': df_accuracy_gain,
        'homophily_original': df_homophily_original,
        'homophily_rewired': df_homophily_rewired,
        'homophily_gain': df_homophily_gain,
        'edge_count_original': df_edge_count_original,
        'edge_count_rewired': df_edge_count_rewired,
        'edge_ratio': df_edge_ratio,
        'summary': df_summary
    }

# Create the dataframes
dataframes = create_comparison_dataframes()

# Display the dataframes
if dataframes:
    print("Accuracy Comparison (Original vs. Rewired)")
    display(dataframes['accuracy_original'])
    display(dataframes['accuracy_rewired'])
    
    print("\nAccuracy Gain (Rewired - Original)")
    display(dataframes['accuracy_gain'])
    
    print("\nHomophily Comparison (Original vs. Rewired)")
    display(dataframes['homophily_original'])
    display(dataframes['homophily_rewired'])
    
    print("\nHomophily Gain (Rewired - Original)")
    display(dataframes['homophily_gain'])
    
    print("\nEdge Count Comparison")
    display(dataframes['edge_count_original'])
    display(dataframes['edge_count_rewired'])
    display(dataframes['edge_ratio'])
    
    print("\nSummary of Average Gains by Model")
    display(dataframes['summary'])
    
    # Create visualizations
    plt.figure(figsize=(12, 8))
    sns.heatmap(dataframes['accuracy_gain'], annot=True, cmap='RdYlGn', center=0)
    plt.title('Accuracy Gain from Improved Attention Rewiring')
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(dataframes['homophily_gain'], annot=True, cmap='RdYlGn', center=0)
    plt.title('Homophily Gain from Improved Attention Rewiring')
    plt.tight_layout()
    plt.show()
    
    # Bar plot of average gains
    plt.figure(figsize=(10, 6))
    dataframes['summary']['Avg Accuracy Gain'].plot(kind='bar', color='skyblue')
    plt.title('Average Accuracy Gain by Model')
    plt.ylabel('Accuracy Gain')
    plt.tight_layout()
    plt.show()
else:
    print("No data available. Please run experiments first using run_all_experiments.sh")
