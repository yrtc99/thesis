#!/bin/bash

# 綜合運行腳本：結合注意力重連和譜重連方法的實驗
# 支持的模型: GCN, SAGE, GAT, APPNP, GCN2, MixHop, GPRGNN, H2GCN
# 支持的數據集: Cora, CiteSeer, PubMed, Cornell, Texas, Wisconsin, Actor, Chameleon, Squirrel

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
DATASET="Cornell"
MODEL="GCN"
REWIRING="attention"  # attention, spectral, both, none
IMPROVED="true"  # 是否使用改進的注意力重連方法
GPU=0
REPEAT=3
SEED=0

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
        --rewiring)
            REWIRING="$2"
            shift
            shift
            ;;
        --improved)
            IMPROVED="$2"
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
        *)
            echo "未知選項: $1"
            exit 1
            ;;
    esac
done

# 打印運行配置
print_header "運行配置"
echo "數據集: $DATASET"
echo "模型: $MODEL"
echo "重連方法: $REWIRING"
echo "使用改進的注意力重連: $IMPROVED"
echo "GPU: $GPU"
echo "重複次數: $REPEAT"
echo "隨機種子: $SEED"
echo ""

# 運行實驗
print_header "開始運行實驗"

# 1. 使用注意力重連的實驗
if [[ "$REWIRING" == "attention" || "$REWIRING" == "both" ]]; then
    if [[ "$IMPROVED" == "true" ]]; then
        print_subheader "運行 $MODEL 與改進的注意力重連 ($DATASET 數據集)"
        
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
            --rewiring attention \
            --thres_min_deg 5 \
            --thres_min_deg_ratio 1.5 \
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
            --prob_drop_edge 0.1
    else
        print_subheader "運行 $MODEL 與標準注意力重連 ($DATASET 數據集)"
        
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
            --rewiring attention \
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
            --prob_drop_edge 0.0
    fi
    
    if [ $? -eq 0 ]; then
        print_success "注意力重連實驗完成！"
    else
        print_error "注意力重連實驗失敗！"
    fi
fi

# 2. 使用譜重連的實驗
if [[ "$REWIRING" == "spectral" || "$REWIRING" == "both" ]]; then
    print_subheader "運行 $MODEL 與譜重連 ($DATASET 數據集)"
    
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
        --prob_drop_edge 0.0
    
    if [ $? -eq 0 ]; then
        print_success "譜重連實驗完成！"
    else
        print_error "譜重連實驗失敗！"
    fi
fi

# 3. 使用兩種重連方法結合的實驗
if [[ "$REWIRING" == "both" ]]; then
    print_subheader "運行 $MODEL 與雙重重連 ($DATASET 數據集)"
    
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
        --prob_drop_edge 0.0
    
    if [ $? -eq 0 ]; then
        print_success "雙重重連實驗完成！"
    else
        print_error "雙重重連實驗失敗！"
    fi
fi

# 4. 不使用重連的基準實驗
if [[ "$REWIRING" == "none" || "$REWIRING" == "all" ]]; then
    print_subheader "運行 $MODEL 基準實驗 ($DATASET 數據集)"
    
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
        --graph_learn False \
        --k 8 \
        --cat_self True
    
    if [ $? -eq 0 ]; then
        print_success "基準實驗完成！"
    else
        print_error "基準實驗失敗！"
    fi
fi

print_header "所有實驗完成"
