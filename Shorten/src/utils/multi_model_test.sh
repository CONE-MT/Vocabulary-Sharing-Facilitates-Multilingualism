#!/bin/bash

# 定义一个数组
list=(
"/cpfs01/user/yuanshuai/model/llama-7b-hf-ori"
"/cpfs01/user/yuanshuai/model/llama-7b-hf-ori"
"/cpfs01/user/yuanshuai/model/llama-7b-hf-ori"
)

dlc_helper=/cpfs01/shared/NLP-A100/NLP-A100_hdd/jie/multilingual_LLM/src/utils/dlc_helper.py
config=/cpfs01/shared/NLP-A100/NLP-A100_hdd/jie/multilingual_LLM/configs/A100/total_inference8.yaml

# 串行版本
# 使用for循环遍历数组
for element in "${list[@]}"; do
  echo "model path: $element"
  /cpfs01/shared/NLP-A100/NLP-A100_hdd/jie/anaconda3/envs/llm2/bin/python $dlc_helper --cfg $config
done

# 并行版本
# # 使用for循环遍历数组
# for i in "${!list[@]}"; do
#   echo "model path: ${list[$i]}"
#   /cpfs01/shared/NLP-A100/NLP-A100_hdd/jie/anaconda3/envs/llm2/bin/python $dlc_helper --cfg $config > "output_$i.log" 2>&1 &
# done

# # 等待所有后台任务完成
# wait