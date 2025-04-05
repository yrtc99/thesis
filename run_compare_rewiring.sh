#!/bin/bash

# 比較不同重連機制的腳本
# 支持的重連類型: original, attention, spectral
# 支持的模型: GCN, SAGE, GAT, APPNP, GCN2, MixHop, GPRGNN, H2GCN
# 支持的數據集: Cora, CiteSeer, PubMed, Cornell, Texas, Wisconsin, Actor, Chameleon, Squirrel

# 顏色輸出函數
function print_header() {
    echo -e "\033[1;34m==== $1 ====\033[0m"
    echo "==== $1 ====" >> "$LOG_FILE"
}

function print_subheader() {
    echo -e "\033[1;36m--- $1 ---\033[0m"
    echo "--- $1 ---" >> "$LOG_FILE"
}

function print_success() {
    echo -e "\033[1;32m$1\033[0m"
    echo "$1" >> "$LOG_FILE"
}

function print_error() {
    echo -e "\033[1;31m$1\033[0m"
    echo "$1" >> "$LOG_FILE"
}

# 設置默認參數
DATASET="Wisconsin"
MODEL="GCN"
REWIRING_TYPES=("original" "attention" "spectral")
GPU=0
REPEAT=3
SEED=0
RESULTS_DIR="./results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# 解析命令行參數
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --dataset)
            DATASET="$2"
            shift
            shift
            ;;
        --model)
            MODEL="$2"
            shift
            shift
            ;;
        --rewiring_types)
            IFS=',' read -r -a REWIRING_TYPES <<< "$2"
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
        --results_dir)
            RESULTS_DIR="$2"
            shift
            shift
            ;;
        *)
            echo "未知選項: $1"
            exit 1
            ;;
    esac
done

# 創建結果目錄
mkdir -p "$RESULTS_DIR"

# 創建實驗結果摘要文件
SUMMARY_FILE="$RESULTS_DIR/${DATASET}_${MODEL}_summary_${TIMESTAMP}.csv"
echo "Rewiring,Train,Val,Test" > "$SUMMARY_FILE"

# 打印運行配置
LOG_FILE="$RESULTS_DIR/${DATASET}_${MODEL}_${TIMESTAMP}.log"
echo "開始實驗: $(date)" > "$LOG_FILE"

print_header "運行配置"
echo "數據集: $DATASET"
echo "模型: $MODEL"
echo "重連類型: ${REWIRING_TYPES[*]}"
echo "GPU: $GPU"
echo "重複次數: $REPEAT"
echo "隨機種子: $SEED"
echo "結果保存目錄: $RESULTS_DIR"
echo ""

echo "數據集: $DATASET" >> "$LOG_FILE"
echo "模型: $MODEL" >> "$LOG_FILE"
echo "重連類型: ${REWIRING_TYPES[*]}" >> "$LOG_FILE"
echo "GPU: $GPU" >> "$LOG_FILE"
echo "重複次數: $REPEAT" >> "$LOG_FILE"
echo "隨機種子: $SEED" >> "$LOG_FILE"
echo "結果保存目錄: $RESULTS_DIR" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

# 運行實驗
print_header "開始運行實驗"

# 為每種重連類型運行實驗
for rewiring_type in "${REWIRING_TYPES[@]}"; do
    print_subheader "運行 $MODEL 與 $rewiring_type 重連 ($DATASET 數據集)"
    
    # 創建此重連類型的日誌文件
    REWIRING_LOG_FILE="$RESULTS_DIR/${DATASET}_${MODEL}_${rewiring_type}_${TIMESTAMP}.log"
    
    # 運行實驗並同時保存到日誌文件
    python main_ae.py \
        --model $MODEL \
        --num_layer 2 \
        --repeat $REPEAT \
        --num_epoch 200 \
        --lr 0.01 \
        --wd 5e-3 \
        --gpu $GPU \
        --data_dir ../data \
        --dataset $DATASET \
        --moment 1 \
        --hidden 64 \
        --use_center_moment \
        --seed $SEED \
        --graph_learn \
        --rewiring_type $rewiring_type \
        --lr_gl 0.001 \
        --wd_gl 5e-3 \
        --thres_min_deg 3 \
        --thres_min_deg_ratio 1.0 \
        --window [10000,10000] \
        --shuffle [False,False] \
        --drop_last [False,False] \
        --epoch_train_gl 200 \
        --epoch_finetune_gl 30 \
        --seed_gl $SEED \
        --k 8 \
        --cat_self True \
        --prunning True \
        --thres_prunning 0.3 \
        --prob_drop_edge 0.0 2>&1 | tee "$REWIRING_LOG_FILE"
    
    # 提取實驗結果並添加到摘要文件
    TRAIN_ACC=$(grep "\[Average Score\]" "$REWIRING_LOG_FILE" | grep -oP "'train': np\.float64\(\K[0-9.]+" | head -1)
    VAL_ACC=$(grep "\[Average Score\]" "$REWIRING_LOG_FILE" | grep -oP "'val': np\.float64\(\K[0-9.]+" | head -1)
    TEST_ACC=$(grep "\[Average Score\]" "$REWIRING_LOG_FILE" | grep -oP "'test': np\.float64\(\K[0-9.]+" | head -1)
    
    # 如果無法提取數值，嘗試其他格式
    if [ -z "$TRAIN_ACC" ]; then
        TRAIN_ACC=$(grep "\[Average Score\]" "$REWIRING_LOG_FILE" | grep -oP "'train': \K[0-9.]+" | head -1)
        VAL_ACC=$(grep "\[Average Score\]" "$REWIRING_LOG_FILE" | grep -oP "'val': \K[0-9.]+" | head -1)
        TEST_ACC=$(grep "\[Average Score\]" "$REWIRING_LOG_FILE" | grep -oP "'test': \K[0-9.]+" | head -1)
    fi
    
    # 添加到摘要文件
    echo "$rewiring_type,$TRAIN_ACC,$VAL_ACC,$TEST_ACC" >> "$SUMMARY_FILE"
    
    if [ $? -eq 0 ]; then
        print_success "$rewiring_type 重連實驗完成！"
    else
        print_error "$rewiring_type 重連實驗失敗！"
    fi
done

# 運行無重連的基準實驗
print_subheader "運行 $MODEL 基準實驗 (無重連) ($DATASET 數據集)"

# 創建基準實驗的日誌文件
BASELINE_LOG_FILE="$RESULTS_DIR/${DATASET}_${MODEL}_baseline_${TIMESTAMP}.log"

python main_ae.py \
    --model $MODEL \
    --num_layer 2 \
    --repeat $REPEAT \
    --num_epoch 200 \
    --lr 0.01 \
    --wd 5e-3 \
    --gpu $GPU \
    --data_dir ../data \
    --dataset $DATASET \
    --moment 1 \
    --hidden 64 \
    --use_center_moment \
    --seed $SEED 2>&1 | tee "$BASELINE_LOG_FILE"

# 提取基準實驗結果並添加到摘要文件
TRAIN_ACC=$(grep "\[Average Score\]" "$BASELINE_LOG_FILE" | grep -oP "'train': np\.float64\(\K[0-9.]+" | head -1)
VAL_ACC=$(grep "\[Average Score\]" "$BASELINE_LOG_FILE" | grep -oP "'val': np\.float64\(\K[0-9.]+" | head -1)
TEST_ACC=$(grep "\[Average Score\]" "$BASELINE_LOG_FILE" | grep -oP "'test': np\.float64\(\K[0-9.]+" | head -1)

# 如果無法提取數值，嘗試其他格式
if [ -z "$TRAIN_ACC" ]; then
    TRAIN_ACC=$(grep "\[Average Score\]" "$BASELINE_LOG_FILE" | grep -oP "'train': \K[0-9.]+" | head -1)
    VAL_ACC=$(grep "\[Average Score\]" "$BASELINE_LOG_FILE" | grep -oP "'val': \K[0-9.]+" | head -1)
    TEST_ACC=$(grep "\[Average Score\]" "$BASELINE_LOG_FILE" | grep -oP "'test': \K[0-9.]+" | head -1)
fi

# 添加到摘要文件
echo "baseline,$TRAIN_ACC,$VAL_ACC,$TEST_ACC" >> "$SUMMARY_FILE"

if [ $? -eq 0 ]; then
    print_success "基準實驗完成！"
else
    print_error "基準實驗失敗！"
fi

print_header "所有實驗完成"
echo "實驗結果摘要保存在: $SUMMARY_FILE"
echo "詳細日誌保存在: $RESULTS_DIR 目錄"

# 創建結果分析筆記本
NOTEBOOK_FILE="$RESULTS_DIR/${DATASET}_${MODEL}_analysis_${TIMESTAMP}.ipynb"

cat > "$NOTEBOOK_FILE" << 'EOL'
{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "# 讀取實驗結果摘要文件\n",
    "summary_df = pd.read_csv('$SUMMARY_FILE')\n",
    "\n",
    "# 繪製實驗結果圖表\n",
    "plt.figure(figsize=(10, 6))\n",
    "plt.plot(summary_df['Rewiring'], summary_df['Train'], label='Train')\n",
    "plt.plot(summary_df['Rewiring'], summary_df['Val'], label='Val')\n",
    "plt.plot(summary_df['Rewiring'], summary_df['Test'], label='Test')\n",
    "plt.xlabel('Rewiring')\n",
    "plt.ylabel('Accuracy')\n",
    "plt.title('Experiment Results')\n",
    "plt.legend()\n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
EOL
