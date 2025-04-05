#!/bin/bash

# Define colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Define datasets to run
DATASETS=("chameleon" "squirrel" "actor" "flickr" "fb100" "cornell" "texas" "wisconsin")

# Define models to run
MODELS=("GCN" "GAT" "GraphSAGE" "APPNP" "GCNII" "H2GCN")

# Define rewiring methods
REWIRING_METHODS=("none" "attention")

# Create results directory if it doesn't exist
mkdir -p results/comparison

# Function to run experiment and save results
run_experiment() {
    local dataset=$1
    local model=$2
    local rewiring=$3
    local improved=$4
    
    echo -e "${BLUE}Running experiment: Dataset=${dataset}, Model=${model}, Rewiring=${rewiring}, Improved=${improved}${NC}"
    
    # Determine output file name
    if [ "$rewiring" == "none" ]; then
        output_file="results/comparison/${dataset}_${model}_original.txt"
        rewiring_arg=""
    else
        if [ "$improved" == "true" ]; then
            output_file="results/comparison/${dataset}_${model}_${rewiring}_improved.txt"
            rewiring_arg="--rewiring ${rewiring} --improved true"
        else
            output_file="results/comparison/${dataset}_${model}_${rewiring}.txt"
            rewiring_arg="--rewiring ${rewiring}"
        fi
    fi
    
    # Run the experiment
    ./run_rewiring_full.sh --dataset ${dataset} --model ${model} ${rewiring_arg} | tee ${output_file}
    
    echo -e "${GREEN}Completed experiment: Dataset=${dataset}, Model=${model}, Rewiring=${rewiring}, Improved=${improved}${NC}"
    echo "Results saved to ${output_file}"
    echo "------------------------------------"
}

# Main execution
echo -e "${YELLOW}Starting all experiments...${NC}"
echo "Total experiments: $((${#DATASETS[@]} * ${#MODELS[@]} * 2))"
echo "------------------------------------"

# First run all models with original graphs (no rewiring)
for dataset in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
        run_experiment "$dataset" "$model" "none" "false"
    done
done

# Then run all models with improved attention rewiring
for dataset in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
        run_experiment "$dataset" "$model" "attention" "true"
    done
done

echo -e "${YELLOW}All experiments completed!${NC}"
echo "Results are saved in the results/comparison directory"
echo "Run the results.ipynb notebook to analyze the results"
