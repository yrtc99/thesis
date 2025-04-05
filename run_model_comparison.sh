#!/bin/bash

# 腳本：比較不同模型在有無 attention rewiring 下的性能
# 作者：Cascade
# 日期：2025-04-01

# 顏色輸出函數
function print_header() {
    echo -e "\033[1;34m==== $1 ====\033[0m"
}

function print_subheader() {
    echo -e "\033[1;36m--- $1 ---\033[0m"
}

function print_success() {
    echo -e "\033[1;32m$1\033[0m"
}

function print_error() {
    echo -e "\033[1;31m$1\033[0m"
}

# 設置默認參數
DATASET="Wisconsin"
GPU=0
REPEAT=3
SEED=0
MODELS="GCN,GAT,GraphSAGE"

# 解析命令行參數
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --dataset)
            DATASET="$2"
            shift
            shift
            ;;
        --gpu)
            GPU="$2"
            shift
            shift
            ;;
        --repeat)
            REPEAT="$2"
            shift
            shift
            ;;
        --seed)
            SEED="$2"
            shift
            shift
            ;;
        --models)
            MODELS="$2"
            shift
            shift
            ;;
        *)
            echo "未知選項: $1"
            exit 1
            ;;
    esac
done

# 將逗號分隔的模型轉換為數組
IFS=',' read -ra MODELS_ARRAY <<< "$MODELS"

# 創建結果目錄
RESULTS_DIR="./model_comparison_results"
mkdir -p "$RESULTS_DIR"

# 創建結果文件
RESULTS_FILE="$RESULTS_DIR/${DATASET}_comparison_results.csv"
echo "Model,Rewiring,Train_Acc,Val_Acc,Test_Acc" > "$RESULTS_FILE"

print_header "開始模型比較實驗 (數據集: $DATASET)"

# 運行每個模型的實驗
for MODEL in "${MODELS_ARRAY[@]}"; do
    print_header "測試模型: $MODEL"
    
    # 1. 使用 attention rewiring
    print_subheader "運行 $MODEL 與 attention rewiring"
    
    # 運行帶有 attention rewiring 的實驗
    bash run_rewiring_full.sh --dataset "$DATASET" --model "$MODEL" --rewiring "attention" --gpu "$GPU" --repeat "$REPEAT" --seed "$SEED" | tee "$RESULTS_DIR/${MODEL}_attention_output.log"
    
    # 從日誌中提取最佳結果
    BEST_RESULTS=$(grep -A 1 "\[Best Score\]" "$RESULTS_DIR/${MODEL}_attention_output.log" | tail -n 1)
    TRAIN_ACC=$(echo "$BEST_RESULTS" | grep -oP "'train': \K[0-9.]+")
    VAL_ACC=$(echo "$BEST_RESULTS" | grep -oP "'val': \K[0-9.]+")
    TEST_ACC=$(echo "$BEST_RESULTS" | grep -oP "'test': \K[0-9.]+")
    
    # 寫入結果到CSV
    echo "$MODEL,attention,$TRAIN_ACC,$VAL_ACC,$TEST_ACC" >> "$RESULTS_FILE"
    
    # 2. 不使用 rewiring
    print_subheader "運行 $MODEL 基準實驗 (無 rewiring)"
    
    # 運行不帶 rewiring 的實驗
    bash run_rewiring_full.sh --dataset "$DATASET" --model "$MODEL" --rewiring "none" --gpu "$GPU" --repeat "$REPEAT" --seed "$SEED" | tee "$RESULTS_DIR/${MODEL}_none_output.log"
    
    # 從日誌中提取最佳結果
    BEST_RESULTS=$(grep -A 1 "\[Best Score\]" "$RESULTS_DIR/${MODEL}_none_output.log" | tail -n 1)
    TRAIN_ACC=$(echo "$BEST_RESULTS" | grep -oP "'train': \K[0-9.]+")
    VAL_ACC=$(echo "$BEST_RESULTS" | grep -oP "'val': \K[0-9.]+")
    TEST_ACC=$(echo "$BEST_RESULTS" | grep -oP "'test': \K[0-9.]+")
    
    # 寫入結果到CSV
    echo "$MODEL,none,$TRAIN_ACC,$VAL_ACC,$TEST_ACC" >> "$RESULTS_FILE"
    
    print_success "$MODEL 實驗完成"
done

print_header "所有實驗完成"
print_subheader "結果摘要 (數據集: $DATASET)"

# 生成結果摘要
echo -e "\n模型比較結果 (測試集準確率):\n"
echo -e "模型\t\t有Rewiring\t無Rewiring\t提升"
echo -e "------------------------------------------------------"

# 讀取CSV並生成比較表格
while IFS=, read -r model rewiring train val test; do
    if [[ "$model" != "Model" ]]; then  # 跳過標題行
        if [[ "$rewiring" == "attention" ]]; then
            # 存儲attention rewiring的結果
            declare "${model}_attention=$test"
        elif [[ "$rewiring" == "none" ]]; then
            # 存儲無rewiring的結果
            declare "${model}_none=$test"
        fi
    fi
done < "$RESULTS_FILE"

# 打印比較結果
for MODEL in "${MODELS_ARRAY[@]}"; do
    attention_var="${MODEL}_attention"
    none_var="${MODEL}_none"
    
    if [[ -n "${!attention_var}" && -n "${!none_var}" ]]; then
        attention_val=${!attention_var}
        none_val=${!none_var}
        improvement=$(echo "scale=4; ($attention_val - $none_val) * 100" | bc)
        
        printf "%-12s\t%-10s\t%-10s\t%+.2f%%\n" "$MODEL" "${!attention_var}" "${!none_var}" "$improvement"
    fi
done

echo -e "\n詳細結果已保存至: $RESULTS_FILE"
