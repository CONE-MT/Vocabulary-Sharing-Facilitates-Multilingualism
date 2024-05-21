PROJ_PATH=/cpfs01/user/yuanshuai/code/how_multi
BASE_MODEL=llama-7b-hf
DATA_PATH=/cpfs01/shared/NLP-A100/NLP-A100_hdd/feyuan/multilingual_LLM/training_data/parallel/sample_10000/en_ca_bilingual_alpaca.json
LANG_PAIR=en-ca
TRAIN_SIZE=10000


# 设置总的batch size
TOTAL_BATCH_SIZE=128
# 设置GPU的个数
NUM_GPUS=1
# 设置每个GPU上的batch size
BATCH_PER_GPU=16
# 计算梯度累积的个数
ACCUMULATION=$(($TOTAL_BATCH_SIZE / ($NUM_GPUS * $BATCH_PER_GPU)))


# 训练模型
PORT=$(comm -23 <(seq 20000 65535 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)

CPFS_PATH=/cpfs01/user/yuanshuai/
MODEL_PATH=/cpfs01/shared/NLP-A100/NLP-A100_hdd/feyuan/model/llama_family
# MODEL_PATH=/cpfs01/shared/NLP-A100/NLP-A100_hdd/jie/base_models/llama-7b-hf
DATASET=$(basename ${DATA_PATH})


OUTPUT_NAME=$BASE_MODEL.$TRAIN_SIZE.$DATASET.full_ft

/cpfs01/shared/NLP-A100/NLP-A100_hdd/jie/anaconda3/envs/llm2/bin/torchrun --nproc_per_node=$NUM_GPUS --master_port=$PORT ${PROJ_PATH}/run_clm_full_ft.py \
    --deepspeed ${PROJ_PATH}/deepspeed_config_zero2.json \
    --model_name_or_path ${MODEL_PATH}/$BASE_MODEL \
    --train_file ${DATA_PATH} \
    --preprocessing_num_workers $NUM_GPUS \
    --dataloader_num_workers 16 \
    --dataloader_pin_memory True \
    --per_device_train_batch_size $BATCH_PER_GPU \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps $ACCUMULATION \
    --save_strategy "steps" \
    --save_steps 99999 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
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
    --cache_dir /cpfs01/shared/NLP-A100/NLP-A100_hdd/jie/finetuned/exps_embed_part/.cache \
    --output_dir ${PROJ_PATH}/models/$OUTPUT_NAME