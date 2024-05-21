# source /cpfs01/shared/NLP-A100/NLP-A100_hdd/feyuan/envs/anaconda3/bin/activate /cpfs01/user/yuanfei/envs/alpaca_env

export http_proxy=http://58.34.83.134:31280/
export https_proxy=http://58.34.83.134:31280/
export HTTP_PROXY=http://58.34.83.134:31280/
export HTTPS_PROXY=http://58.34.83.134:31280/

# export NCCL_IB_TC=136
# export NCCL_IB_SL=5
# export NCCL_IB_GID_INDEX=3
# export NCCL_SOCKET_IFNAME=bond0
# export NCCL_DEBUG=INFO
# export NCCL_IB_HCA=mlx5
# export NCCL_IB_TIMEOUT=22
# export NCCL_IB_QPS_PER_CONNECTION=8
# export NCCL_NET_PLUGIN=none
PROJ_PATH=/cpfs01/user/yuanshuai/code/how_multi

# 设置总的batch size
TOTAL_BATCH_SIZE=128
# 设置GPU的个数
NUM_GPUS=1
# 设置每个GPU上的batch size
BATCH_PER_GPU=16
# 计算梯度累积的个数
ACCUMULATION=$(($TOTAL_BATCH_SIZE / ($NUM_GPUS * $BATCH_PER_GPU)))



PROJ_PATH=/cpfs01/user/yuanshuai/code/how_multi
MODEL_PATH=/cpfs01/shared/NLP-A100/NLP-A100_hdd/jie/base_models/llama-7b-hf
DATA_PATH=/cpfs01/shared/NLP-A100/NLP-A100_hdd/feyuan/multilingual_LLM/training_data/parallel/sample_10000/en_ca_bilingual_alpaca.json
OUTPUT_DIR=${PROJ_PATH}/models/lora

PORT=$(comm -23 <(seq 20000 65535 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)

# CPFS_PATH=/cpfs01/shared/NLP-A100/NLP-A100_hdd/feyuan
DATASET=$(basename ${DATA_PATH})

/cpfs01/shared/NLP-A100/NLP-A100_hdd/jie/anaconda3/envs/llm2/bin/torchrun --nproc_per_node=$NUM_GPUS --master_port=$PORT ${PROJ_PATH}/run_clm_lora_ft.py \
    --deepspeed ${PROJ_PATH}/deepspeed_config_zero2.json \
    --model_name_or_path ${MODEL_PATH} \
    --train_file ${DATA_PATH} \
    --cache_dir $(dirname ${DATA_PATH})/.cache \
    --preprocessing_num_workers $NUM_GPUS \
    --dataloader_num_workers 16 \
    --dataloader_pin_memory True \
    --per_device_train_batch_size $BATCH_PER_GPU \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps $ACCUMULATION \
    --save_strategy "steps" \
    --save_steps 99999 \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --block_size 1024 \
    --do_train \
    --evaluation_strategy "no" \
    --num_train_epochs 3 \
    --validation_split_percentage 0 \
    --fp16 True \
    --fp16_full_eval True \
    --streaming \
    --ddp_timeout 3600 \
    --seed 42 \
    --gradient_checkpointing True \
    --output_dir ${OUTPUT_DIR} \
    --use_lora True \
    --lora_config /cpfs01/shared/NLP-A100/NLP-A100_hdd/jie/multilingual_LLM/scripts/trans.train/batch_train_chatglm/lora_config_chatglm.json
