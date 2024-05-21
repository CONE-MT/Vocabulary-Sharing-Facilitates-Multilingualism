import argparse
import logging
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, AutoConfig
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import os
import json
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
        
        model = ChatGLMForConditionalGeneration.from_pretrained(config.model_path.base_model, 
                                                     torch_dtype=torch_dtype_dict[config.model_path.torch_dtype],trust_remote_code=True,
                                                     device_map="auto")
        if config.model_path.lora is not None:
            model = PeftModel.from_pretrained(model, config.model_path.lora, 
                                              torch_dtype=torch_dtype_dict[config.model_path.torch_dtype],trust_remote_code=True,
                                             )
        print(model.hf_device_map)
        # bloom uses only fast tokenize
        to_use_fast = False
        if "bloom" in model_name_or_path or "mpt" in model_name_or_path:
            to_use_fast = True

        tokenizer = ChatGLMTokenizer.from_pretrained(config.model_path.base_model,  use_fast=to_use_fast,trust_remote_code=True)
    else:
        model_name_or_path = config.model_path.base_model
        model = AutoModelForCausalLM.from_pretrained(config.model_path.base_model, 
                                                     torch_dtype=torch_dtype_dict[config.model_path.torch_dtype],trust_remote_code=True,
                                                     device_map="auto")
        if config.model_path.lora is not None:
            model = PeftModel.from_pretrained(model, config.model_path.lora, 
                                              torch_dtype=torch_dtype_dict[config.model_path.torch_dtype],trust_remote_code=True,
                                             )
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
    num_beams = config.generat.num_beams
    temperature = config.generat.temperature
    # output_file = config.output.output_file
    if config.dataset.labels is not None: #["A", "B", "C"]
        # force_words_ids = [tokenizer(["A", "B", "C"], add_special_tokens=False).input_ids]
        force_words_ids = [[29909, 29933, 29907]] #force_words_ids[0]
        print(f"采用了force word ids: {config.dataset.labels}, 对应的token id是{force_words_ids}")
    # gen_config = GenerationConfig(temperature=temperature,
    #                               top_p=0.9,
    #                               do_sample=False,
    #                               num_beams=num_beams,
    #                               max_new_tokens=1,
    #                               eos_token_id=tokenizer.eos_token_id,
    #                               pad_token=tokenizer.pad_token_id,
    #                               force_word_ids=[tokenizer(config.dataset.labels, add_special_tokens=False)[
    #                                                   "input_ids"]] if config.dataset.labels else None
    #                               # bad_words_ids=[tokenizer("\n", add_special_tokens=False)[
    #                               #                     "input_ids"]]
    #                               )
    # 上面是错的
    
    gen_config = GenerationConfig(temperature=0.7,
                                  top_p=0.9,
                                  do_sample=False,
                                  num_beams=num_beams,
                                  max_new_tokens=1,
                                  eos_token_id=tokenizer.eos_token_id,
                                  pad_token=tokenizer.pad_token_id,
                                  force_words_ids=force_words_ids if config.dataset.labels else None,
                                  # force_word_ids=[tokenizer(config.dataset.labels, add_special_tokens=False)[
                                  #                     "input_ids"]] if config.dataset.labels else None
                                  # bad_words_ids=[tokenizer("\n", add_special_tokens=False)[
                                  #                     "input_ids"]]
                                  )

    return model, tokenizer, gen_config


def load_generat_config(config, tokenizer):
    
    batch_size = config.generat.batch_size
    num_beams = config.generat.num_beams
    temperature = config.generat.temperature
    gen_config = GenerationConfig(temperature=0.7,
                                  top_p=0.9,
                                  do_sample=False,
                                  num_beams=num_beams,
                                  max_new_tokens=1,
                                  eos_token_id=tokenizer.eos_token_id,
                                  pad_token=tokenizer.pad_token_id,
                                  force_words_ids=force_words_ids if config.dataset.labels else None,
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

def my_accuracy(targets, preds):
    right_count = 0.
    count = 0.
    for t, p in zip(targets, preds):
        print(t, p)
        if t==p:
            right_count+=1
        count +=1
    return right_count/count

def force_word_func(socre, hand_force_ids, tokenizer, num_beams):
    assert num_beams==1
#     # 在第0维度上，每num_beams个分一组
#     socre_split = socre.split(num_beams, dim=0)
#     print(socre_split)

#     # 对于每组，取第二维度的max
#     max_values = [torch.max(sub_x, dim=0).values for sub_x in socre_split]
    
#     # 将max_values转换为Tensor
#     socre = torch.stack(max_values, dim=0)
#     print(socre.size())
    # force_ids = hand_force_ids #tokenizer(labels, add_special_tokens=False)["input_ids"]
    # print(force_ids,socre.size())
    # print(torch.argmax(socre[:,force_ids],dim=-1))
    # print(socre[:,hand_force_ids].size())
    final_ids = []
    for max_score_id in torch.argmax(socre[:,hand_force_ids],dim=-1):
        final_ids.append(hand_force_ids[max_score_id])
    # final_id = force_ids[torch.argmax(socre[0,force_ids])]
    res=tokenizer.batch_decode(final_ids, skip_special_tokens=True)
    return res

def test_process(config, data_dict):
    batch_size = config.generat.batch_size
    print(f"加载模型...")
    model, tokenizer, gen_config = load_model(config)
    print(f"模型加载完成")
    
    print(f"开始生成...")
    input_data = data_dict["input_data"]
    prompt_list = data_dict["prompt_list"]
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
    # Generate
    targets = [data[-1] for data in input_data]
    predictions = []
    # path_split = output_file.split('.')
    # output_raw_file = '.'.join(path_split[:-1] + ['raw'] + ['txt'])
    # os.makedirs(os.path.dirname(output_file), exist_ok=True)
    # force_words_ids = [[[319], [350], [315]]]#[[319, 350, 315]]#[[319], [350], [315]] #tokenizer(["A B C"], add_special_tokens=False).input_ids
    
    print(f"样本个数：{len(prompt_list)}, {len(input_data)}")
    with open(output_file, 'w', encoding='utf-8') as f, open(output_raw_file, 'w') as g:
        # for i in tqdm(range(0, 50, batch_size)):
        for i in tqdm(range(0, len(prompt_list), batch_size)):
            p = prompt_list[i:i + batch_size]
            tokenized = tokenizer(p, padding=True, return_tensors="pt")
            input_ids = tokenized.input_ids.cuda()
            attn_mask = tokenized.attention_mask.cuda()
            input_ids = input_ids[:, :-1] if input_ids[0, -1] == tokenizer.eos_token_id else input_ids
            attn_mask = attn_mask[:, :-1] if input_ids[0, -1] == tokenizer.eos_token_id else attn_mask

            with torch.no_grad():
                generation_output = model.generate(inputs=input_ids,
                                               attention_mask=attn_mask,
                                               generation_config=gen_config,
                                               pad_token_id=tokenizer.eos_token_id,
                                               return_dict_in_generate=True,
                                                output_scores=True,)

            decoded_tokens = tokenizer.batch_decode(generation_output.sequences, skip_special_tokens=False)
            if config.generat.force_word_ids == 'oj':
                # print(generation_output.scores[-1].size())
                oj_preds = force_word_func(generation_output["scores"][-1], hand_force_ids=config.model_path.hand_force_ids, tokenizer=tokenizer, num_beams=config.generat.num_beams)
            
            for i, dec in enumerate(decoded_tokens):
                if config.generat.force_word_ids == 'oj':
                    prediction = oj_preds[i].strip()
                else:
                    prediction = post_process(dec)
                predictions.append(prediction)
                
                # f.write(prediction+" "+ str(generated_ids[i,...][input_ids[i,...].size(0):]))
                f.write(prediction)
                f.write('\n')

                g.write(dec)
                g.write('\n')
                g.write('\n')
            # sys.exit(0)
        accuracy = accuracy_score(targets[:len(predictions)], predictions)
        # accuracy_info = 'accuracy: {}'.format(float(accuracy)*100)
        print(f"accuracy: {accuracy*100}")
        # print(f"my acc： {my_accuracy(targets[:len(predictions)], predictions)}")
        # f.write(accuracy_info)
        f.writelines(
            json.dumps({"accuracy": accuracy * 100}, ensure_ascii=False) + "\n" 
        )
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
    # Generate
    targets = [data[-1] for data in input_data]
    predictions = []
    # path_split = output_file.split('.')
    # output_raw_file = '.'.join(path_split[:-1] + ['raw'] + ['txt'])
    # os.makedirs(os.path.dirname(output_file), exist_ok=True)
    # force_words_ids = [[[319], [350], [315]]]#[[319, 350, 315]]#[[319], [350], [315]] #tokenizer(["A B C"], add_special_tokens=False).input_ids
    
    print(f"样本个数：{len(prompt_list)}, {len(input_data)}")
    with open(output_file, 'w', encoding='utf-8') as f, open(output_raw_file, 'w') as g:
        # for i in tqdm(range(0, 50, batch_size)):
        for i in tqdm(range(0, len(prompt_list), batch_size)):
            p = prompt_list[i:i + batch_size]
            tokenized = tokenizer(p, padding=True, return_tensors="pt")
            input_ids = tokenized.input_ids.cuda()
            attn_mask = tokenized.attention_mask.cuda()
            input_ids = input_ids[:, :-1] if input_ids[0, -1] == tokenizer.eos_token_id else input_ids
            attn_mask = attn_mask[:, :-1] if input_ids[0, -1] == tokenizer.eos_token_id else attn_mask

            with torch.no_grad():
                generation_output = model.generate(inputs=input_ids,
                                               attention_mask=attn_mask,
                                               generation_config=gen_config,
                                               pad_token_id=tokenizer.eos_token_id,
                                               return_dict_in_generate=True,
                                                output_scores=True,)

            decoded_tokens = tokenizer.batch_decode(generation_output.sequences, skip_special_tokens=False)
            if config.generat.force_word_ids == 'oj':
                # print(generation_output.scores[-1].size())
                oj_preds = force_word_func(generation_output["scores"][-1], hand_force_ids=config.model_path.hand_force_ids, tokenizer=tokenizer, num_beams=config.generat.num_beams)
            
            for i, dec in enumerate(decoded_tokens):
                if config.generat.force_word_ids == 'oj':
                    prediction = oj_preds[i].strip()
                else:
                    prediction = post_process(dec)
                predictions.append(prediction)
                
                # f.write(prediction+" "+ str(generated_ids[i,...][input_ids[i,...].size(0):]))
                f.write(prediction)
                f.write('\n')

                g.write(dec)
                g.write('\n')
                g.write('\n')
            # sys.exit(0)
        accuracy = accuracy_score(targets[:len(predictions)], predictions)
        # accuracy_info = 'accuracy: {}'.format(float(accuracy)*100)
        print(f"accuracy: {accuracy*100}")
        # print(f"my acc： {my_accuracy(targets[:len(predictions)], predictions)}")
        # f.write(accuracy_info)
        f.writelines(
            json.dumps({"accuracy": accuracy * 100}, ensure_ascii=False) + "\n" 
        )
    print("Finished!")