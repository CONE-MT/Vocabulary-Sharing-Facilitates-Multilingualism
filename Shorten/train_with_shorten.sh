#这个脚本是为了baseline：不扩词表直接FT
PROJ_PATH=/cpfs01/user/yuanshuai/code/how_multi
TRAIN_SIZE=10000
SRC_LANG=en
TGT_LANG=te
BASE_MODEL=llama-7b-hf
DATA_PATH=/cpfs01/shared/NLP-A100/NLP-A100_hdd/feyuan/multilingual_LLM/training_data/parallel/sample_$TRAIN_SIZE/${SRC_LANG}_${TGT_LANG}_bilingual_alpaca.json
METHOD=compress
SETTING=single_227
LANG_PAIR=en-te


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

OUTPUT_NAME=$BASE_MODEL.$TRAIN_SIZE.$DATASET.$METHOD.$SETTING
# basemodel
# dataset=（wiki，lego， +量）
# full-ft, lora-ft
/cpfs01/shared/NLP-A100/NLP-A100_hdd/jie/anaconda3/envs/llm2/bin/torchrun --nproc_per_node=$NUM_GPUS --master_port=$PORT  ${PROJ_PATH}/Shorten/run_clm_shorten.py \
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
    --save_steps 999999 \
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
    --output_dir /cpfs01/user/yuanshuai/code/tokenize/finetuned/compress/$OUTPUT_NAME \
    --shorten_id 227
    #/cpfs01/user/yuanshuai/code/tokenize/debug/
    #   --compress_map /cpfs01/shared/NLP-A100/NLP-A100_hdd/feyuan/data/lego_raw_data/${LANG_PAIR}/analysis/rm2_map.json \
 
# 推理测试
# bash /cpfs01/user/yuanshuai/code/tokenize/batch_train/test_after_train.sh $TEST_MODEL_PAT

# /cpfs01/shared/NLP-A100/NLP-A100_hdd/jie/anaconda3/envs/llm2/bin/python /cpfs01/shared/NLP-A100/NLP-A100_hdd/jie/multilingual_LLM/src/utils/dlc_helper.py --cfg /cpfs01/user/yuanshuai/code/tokenize/config/inference.yaml --model_path /cpfs01/user/yuanshuai/code/tokenize/finetuned/$OUTPUT_NAME

# cd /cpfs01/shared/NLP-A100/NLP-A100_hdd/jie/multilingual_LLM/

# /cpfs01/shared/NLP-A100/NLP-A100_hdd/jie/anaconda3/envs/llm2/bin/python /cpfs01/shared/NLP-A100/NLP-A100_hdd/jie/multilingual_LLM/eval.py --cfg /cpfs01/shared/NLP-A100/NLP-A100_hdd/jie/multilingual_LLM/configs/A100/flores.yaml --input_file eng.devtest --lang_pair $LANG_PAIR --base_model /cpfs01/user/yuanshuai/code/tokenize/finetuned/$OUTPUT_NAME  --beam_size 4 --project_path /cpfs01/shared/NLP-A100/NLP-A100_hdd/jie/multilingual_LLM/