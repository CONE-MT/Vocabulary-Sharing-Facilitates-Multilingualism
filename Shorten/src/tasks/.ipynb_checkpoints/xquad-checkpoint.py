import argparse
import logging
import json
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, AutoConfig
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import os
import pandas as pd
from peft import PeftModel
import sys
sys.path.append("/cpfs01/shared/NLP-A100/NLP-A100_hdd/jie/multilingual_LLM/scripts/trans.train/batch_train_chatglm/chatglm2-6b/")
from tokenization_chatglm import ChatGLMTokenizer
from modeling_chatglm import ChatGLMForConditionalGeneration

DEFAULT_PAD_TOKEN = "</s>"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

def load_model(config):
    # Load checkpoints
    torch_dtype_dict = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32
    }
    model_name_or_path = config.model_path.base_model
    if 'chatglm' in model_name_or_path:
        
        model = ChatGLMForConditionalGeneration.from_pretrained(config.model_path.base_model, torch_dtype=torch_dtype_dict[config.model_path.torch_dtype], trust_remote_code=True, device_map="auto")
        if config.model_path.lora is not None:
            model = PeftModel.from_pretrained(model, config.model_path.lora, torch_dtype=torch_dtype_dict[config.model_path.torch_dtype],trust_remote_code=True)
        print(model.hf_device_map)
        # bloom uses only fast tokenize
        to_use_fast = False
        if "bloom" in model_name_or_path or "mpt" in model_name_or_path:
            to_use_fast = True

        tokenizer = ChatGLMTokenizer.from_pretrained(config.model_path.base_model,  use_fast=to_use_fast,trust_remote_code=True)
    else:
        model_name_or_path = config.model_path.base_model
        model = AutoModelForCausalLM.from_pretrained(config.model_path.base_model, torch_dtype=torch_dtype_dict[config.model_path.torch_dtype], trust_remote_code=True, device_map="auto")
        if config.model_path.lora is not None:
            model = PeftModel.from_pretrained(model, config.model_path.lora, torch_dtype=torch_dtype_dict[config.model_path.torch_dtype],trust_remote_code=True)
        print(model.hf_device_map)
        # bloom uses only fast tokenize
        to_use_fast = False
        if "bloom" in model_name_or_path or "mpt" in model_name_or_path:
            to_use_fast = True

        tokenizer = AutoTokenizer.from_pretrained(config.model_path.base_model,  use_fast=to_use_fast,trust_remote_code=True)
    tokenizer.padding_side = "left"

    # if tokenizer.pad_token is None:
    #     tokenizer.pad_token = tokenizer.eos_token
    #     tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.pad_token is None and "mpt" not in model_name_or_path:
        print("添加，特殊token")
        tokenizer.add_special_tokens(
            {
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
                "pad_token": tokenizer.eos_token,
            }
        )
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.pad_token is None and "mpt" in model_name_or_path:
        tokenizer.add_special_tokens(
            {
                "pad_token": "<|endoftext|>",
                "eos_token": "<|endoftext|>"
            }
        )
        tokenizer.pad_token_id = tokenizer.eos_token_id
        # model.config.pad_token_id = tokenizer.pad_token_id
        print("mpt 添加，特殊token；模型也得加")

    print("展示下，tokenizer的特殊符号id：",tokenizer.eos_token_id,tokenizer.pad_token_id, tokenizer.eos_token, tokenizer.pad_token)
        
    batch_size = config.generat.batch_size
    beam_size = config.generat.beam_size
    temperature = config.generat.temperature
    # output_file = config.output.output_file

    gen_config = GenerationConfig(temperature=temperature,
                                  top_p=0.9,
                                  do_sample=False,
                                  num_beams=beam_size,
                                  max_new_tokens=256,
                                  eos_token_id=tokenizer.eos_token_id,
                                  pad_token=tokenizer.pad_token_id,
                                  )

    
    return model, tokenizer, gen_config


def load_generat_config(config, tokenizer):
    
    batch_size = config.generat.batch_size
    beam_size = config.generat.beam_size
    temperature = config.generat.temperature
    # output_file = config.output.output_file

    gen_config = GenerationConfig(temperature=temperature,
                                  top_p=0.9,
                                  do_sample=False,
                                  num_beams=beam_size,
                                  max_new_tokens=256,
                                  eos_token_id=tokenizer.eos_token_id,
                                  pad_token=tokenizer.pad_token_id,
                                  )
    return gen_config


# Post-process the output, extract translations
def post_process(text):
    text = text.split("### Response:")[1].strip()
    text = text.replace("\n", " ")
    # Cut for contrastive instruction
    if "</p>" in text:
        text = text.split("</p>")[0].split("<p>")[-1]
    return text

def test_process(config, data_dict):
    batch_size = config.generat.batch_size
    print(f"加载模型...")
    model, tokenizer, gen_config = load_model(config)
    print(f"模型加载完成")
    
    print(f"开始生成...")
    input_data = data_dict["input_data"]
    prompt_list = data_dict["prompt_list"]
    id_list = data_dict["id_list"]
    print(f"准备结果记录日志...")
    
    record_base_path = config.model_path.lora if config.model_path.lora else config.model_path.base_model
    dataset_name = config.dataset.loader
    lang_pair_name = config.dataset.lang_pair if config.dataset.lang_pair else ""
    record_dir=os.path.join(record_base_path,dataset_name)
    os.makedirs(record_dir, exist_ok=True)
    record_lang_dir=os.path.join(record_base_path,dataset_name,lang_pair_name)
    os.makedirs(record_lang_dir, exist_ok=True)
    
    
    output_file = os.path.join(record_lang_dir, config.output.output_file_prefix+".txt")
    output_raw_file = os.path.join(record_lang_dir, config.output.output_file_prefix+"_raw.txt")
    # if os.path.exists(output_file):
    #     sys.exit(0)
    
    
    # Generate
    predictions = {}
    predictions_raw = []
    # path_split = output_file.split('.')
    # output_raw_file = '.'.join(path_split[:-1] + ['raw'] + ['txt'])

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    

    with open(output_file, 'w', encoding='utf-8') as f, open(output_raw_file, 'w') as g:
        # for i in tqdm(range(0, 100, batch_size)):
        for i in tqdm(range(0, len(prompt_list), batch_size)):
            p = prompt_list[i:i+batch_size]
            ids = id_list[i:i+batch_size]
            tokenized = tokenizer(p, padding=True, return_tensors="pt")
            input_ids = tokenized.input_ids.cuda()
            attn_mask = tokenized.attention_mask.cuda()
            input_ids = input_ids[:, :-1] if input_ids[0, -1] == tokenizer.eos_token_id else input_ids
            attn_mask = attn_mask[:, :-1] if input_ids[0, -1] == tokenizer.eos_token_id else attn_mask

            with torch.no_grad():
                generated_ids = model.generate(inputs=input_ids,
                                               attention_mask=attn_mask,
                                               generation_config=gen_config,
                                               pad_token_id=tokenizer.eos_token_id)

            decoded_tokens = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            for dec, id in zip(decoded_tokens, ids):
                predictions[id] = post_process(dec)
                g.write(dec)
                g.write('\n')
                g.write('\n')

                # predictions_raw.append(dec+'/n')

        jsoned = json.dumps(predictions, ensure_ascii=False)
        f.write(jsoned)
        f.write('\n')

    # python evaluate_squad.py --dataset-file '/mnt/petrelfs/zhuwenhao/experiments/raw-data/xquad/xquad.en.json'
    # --prediction-file '/mnt/petrelfs/zhuwenhao/experiments/checkpoints/llm/llama.wmt17-en2es-100k-alpaca/generation_results.xquad_en.json'
    metric_file_path = os.path.join(os.path.dirname(output_file), "metric.json")
    # abspath = os.path.abspath("./")
    intput_file_path=os.path.join(config.dataset.path, config.dataset.input_file+f".{config.dataset.lang_pair}.json")
    # metric_file_path=config.metric.metric_json
    eval_python_script=config.metric.evalution
    if config.metric.python_path:
        command = f"{config.metric.python_path} {eval_python_script} --dataset-file {intput_file_path} --prediction-file {output_file} --metric-file {metric_file_path}"
    else:
        command = f"python {eval_python_script} --dataset-file {intput_file_path} --prediction-file {output_file} --metric-file {metric_file_path}"
    print(f"获取结果的命令如下：{command}")
    os.system(command)
    print("Finished!")
    
    
def test_process_no_model_load(config, data_dict, model, tokenizer):
    batch_size = config.generat.batch_size
    # print(f"加载模型...")
    # model, tokenizer, gen_config = load_model(config)
    # print(f"模型加载完成")
    gen_config = load_generat_config(config, tokenizer)
    
    print(f"开始生成...")
    input_data = data_dict["input_data"]
    prompt_list = data_dict["prompt_list"]
    id_list = data_dict["id_list"]
    print(f"准备结果记录日志...")
    
    record_base_path = config.model_path.lora if config.model_path.lora else config.model_path.base_model
    dataset_name = config.dataset.loader
    lang_pair_name = config.dataset.lang_pair if config.dataset.lang_pair else ""
    record_dir=os.path.join(record_base_path,dataset_name)
    os.makedirs(record_dir, exist_ok=True)
    record_lang_dir=os.path.join(record_base_path,dataset_name,lang_pair_name)
    os.makedirs(record_lang_dir, exist_ok=True)
    
    
    output_file = os.path.join(record_lang_dir, config.output.output_file_prefix+".txt")
    output_raw_file = os.path.join(record_lang_dir, config.output.output_file_prefix+"_raw.txt")
    # if os.path.exists(output_file):
    #     sys.exit(0)
    
    
    # Generate
    predictions = {}
    predictions_raw = []
    # path_split = output_file.split('.')
    # output_raw_file = '.'.join(path_split[:-1] + ['raw'] + ['txt'])

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    

    with open(output_file, 'w', encoding='utf-8') as f, open(output_raw_file, 'w') as g:
        # for i in tqdm(range(0, 100, batch_size)):
        for i in tqdm(range(0, len(prompt_list), batch_size)):
            p = prompt_list[i:i+batch_size]
            ids = id_list[i:i+batch_size]
            tokenized = tokenizer(p, padding=True, return_tensors="pt")
            input_ids = tokenized.input_ids.cuda()
            attn_mask = tokenized.attention_mask.cuda()
            input_ids = input_ids[:, :-1] if input_ids[0, -1] == tokenizer.eos_token_id else input_ids
            attn_mask = attn_mask[:, :-1] if input_ids[0, -1] == tokenizer.eos_token_id else attn_mask

            with torch.no_grad():
                generated_ids = model.generate(inputs=input_ids,
                                               attention_mask=attn_mask,
                                               generation_config=gen_config,
                                               pad_token_id=tokenizer.eos_token_id)

            decoded_tokens = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            for dec, id in zip(decoded_tokens, ids):
                predictions[id] = post_process(dec)
                g.write(dec)
                g.write('\n')
                g.write('\n')

                # predictions_raw.append(dec+'/n')

        jsoned = json.dumps(predictions, ensure_ascii=False)
        f.write(jsoned)
        f.write('\n')

    # python evaluate_squad.py --dataset-file '/mnt/petrelfs/zhuwenhao/experiments/raw-data/xquad/xquad.en.json'
    # --prediction-file '/mnt/petrelfs/zhuwenhao/experiments/checkpoints/llm/llama.wmt17-en2es-100k-alpaca/generation_results.xquad_en.json'
    metric_file_path = os.path.join(os.path.dirname(output_file), "metric.json")
    # abspath = os.path.abspath("./")
    intput_file_path=os.path.join(config.dataset.path, config.dataset.input_file+f".{config.dataset.lang_pair}.json")
    # metric_file_path=config.metric.metric_json
    eval_python_script=config.metric.evalution
    if config.metric.python_path:
        command = f"{config.metric.python_path} {eval_python_script} --dataset-file {intput_file_path} --prediction-file {output_file} --metric-file {metric_file_path}"
    else:
        command = f"python {eval_python_script} --dataset-file {intput_file_path} --prediction-file {output_file} --metric-file {metric_file_path}"
    print(f"获取结果的命令如下：{command}")
    os.system(command)
    print("Finished!")