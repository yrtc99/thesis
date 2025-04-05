#!/usr/bin/env python3
"""
Script to analyze the results of the rewiring experiments and generate DataFrames
for comparison between original and improved attention rewiring methods.
"""

import os
import re
import glob
import pandas as pd
import numpy as np

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
    accuracy_original = {model: {} for model in MODELS}
    accuracy_rewired = {model: {} for model in MODELS}
    homophily_original = {model: {} for model in MODELS}
    homophily_rewired = {model: {} for model in MODELS}
    edge_count_original = {model: {} for model in MODELS}
    edge_count_rewired = {model: {} for model in MODELS}
    
    # Process all result files
    results_dir = "results/comparison"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"Created directory {results_dir}. Please run experiments first.")
        return
    
    for dataset in DATASETS:
        for model in MODELS:
            # Original results
            original_file = f"{results_dir}/{dataset}_{model}_original.txt"
            if os.path.exists(original_file):
                metrics = extract_metrics_from_file(original_file)
                accuracy_original[model][dataset] = metrics['accuracy']
                homophily_original[model][dataset] = metrics['original_homophily']
                edge_count_original[model][dataset] = metrics['original_edge_count']
            
            # Rewired results
            rewired_file = f"{results_dir}/{dataset}_{model}_attention_improved.txt"
            if os.path.exists(rewired_file):
                metrics = extract_metrics_from_file(rewired_file)
                accuracy_rewired[model][dataset] = metrics['accuracy']
                homophily_rewired[model][dataset] = metrics['new_homophily']
                edge_count_rewired[model][dataset] = metrics['new_edge_count']
    
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
    
    return {
        'accuracy_original': df_accuracy_original,
        'accuracy_rewired': df_accuracy_rewired,
        'accuracy_gain': df_accuracy_gain,
        'homophily_original': df_homophily_original,
        'homophily_rewired': df_homophily_rewired,
        'homophily_gain': df_homophily_gain,
        'edge_count_original': df_edge_count_original,
        'edge_count_rewired': df_edge_count_rewired,
        'edge_ratio': df_edge_ratio
    }

def main():
    """Main function to create and display comparison DataFrames."""
    # Check if results directory exists
    if not os.path.exists("results/comparison"):
        print("Results directory not found. Please run experiments first using run_all_experiments.sh")
        return
    
    # Create comparison DataFrames
    dataframes = create_comparison_dataframes()
    
    # Print DataFrames
    print("\n===== ACCURACY COMPARISON =====")
    print("\nOriginal Accuracy:")
    print(dataframes['accuracy_original'].T)
    
    print("\nRewired Accuracy:")
    print(dataframes['accuracy_rewired'].T)
    
    print("\nAccuracy Gain:")
    print(dataframes['accuracy_gain'].T)
    
    print("\n===== HOMOPHILY COMPARISON =====")
    print("\nOriginal Homophily:")
    print(dataframes['homophily_original'].T)
    
    print("\nRewired Homophily:")
    print(dataframes['homophily_rewired'].T)
    
    print("\nHomophily Gain:")
    print(dataframes['homophily_gain'].T)
    
    print("\n===== EDGE COUNT COMPARISON =====")
    print("\nOriginal Edge Count:")
    print(dataframes['edge_count_original'].T)
    
    print("\nRewired Edge Count:")
    print(dataframes['edge_count_rewired'].T)
    
    print("\nEdge Count Ratio (Rewired/Original):")
    print(dataframes['edge_ratio'].T)
    
    # Save DataFrames to CSV
    for name, df in dataframes.items():
        df.to_csv(f"results/comparison/{name}.csv")
    
    print("\nDataFrames saved to CSV files in results/comparison directory")

if __name__ == "__main__":
    main()
