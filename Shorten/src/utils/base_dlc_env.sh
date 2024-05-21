# 示例命令
# bash dlc_flores101.sh model_path input_file save_file language_pair

# source /cpfs01/shared/NLP-A100/NLP-A100_hdd/feyuan/envs/anaconda3/bin/activate /cpfs01/user/yuanfei/envs/alpaca_env


CPFS_PATH=/cpfs01/user/yuanshuai/

bash /cpfs01/shared/NLP-A100/NLP-A100_hdd/jie/multilingual_LLM/src/utils/dlc.sh
# bash /cpfs01/user/yuanfei/envs/dlc_config.sh


# NAME=basic_inference_$4
NAME=inference_c_evalR
# COMMAND="/cpfs01/user/yuanfei/envs/alpaca_env/bin/python  /cpfs01/shared/NLP-A100/NLP-A100_hdd/feyuan/code/multilingual_llm/train/inference.py --model-name-or-path $1 -lp $4 -t 0.1 -sa 'beam' -ins /cpfs01/shared/NLP-A100/NLP-A100_hdd/feyuan/code/multilingual_llm/test/instruct_inf.txt -i $2 -o $3 "

COMMAND="/cpfs01/shared/NLP-A100/NLP-A100_hdd/jie/anaconda3/envs/llm2/bin/python eval.py --cfg /cpfs01/shared/NLP-A100/NLP-A100_hdd/jie/multilingual_LLM/configs/ceval.yaml --data_path /cpfs01/shared/NLP-A100/NLP-A100_hdd/jie/benchmark/c-Eval/ceval_all_zh_zeroshot --lang_pair zh"
echo ${COMMAND}

${CPFS_PATH}/dlc create job \
    --command "$COMMAND" \
    --data_sources "" \
    --kind "PyTorchJob" \
    --name "$NAME" \
    --node_names "" \
    --priority "1" \
    --resource_id "quotactf01wuemor" \
    --thirdparty_lib_dir "" \
    --thirdparty_libs "" \
    --worker_count "1" \
    --worker_cpu "4" \
    --worker_gpu "1" \
    --worker_gpu_type "nvidia_a100-sxm4-80gb" \
    --worker_image "pjlab-wulan-acr-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/pjlab-eflops/yuanshuai:llama-jie-v1" \
    --worker_memory "32Gi" \
    --worker_shared_memory "32Gi" \
    --workspace_id "wsvnint04sctwu5a"