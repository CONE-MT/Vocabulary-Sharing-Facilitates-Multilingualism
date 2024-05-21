import argparse
import logging
import sys
from src.datasets.common import get_translation_from_hyp, get_spBLEU
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, AutoConfig
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import os
import pandas as pd
from peft import PeftModel
import sys


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

    # model_name_or_path = config.model_path.base_model
    model = AutoModelForCausalLM.from_pretrained(config.model_path.base_model, 
                                                 torch_dtype=torch_dtype_dict[config.model_path.torch_dtype], #trust_remote_code=True,
                                                 device_map="auto")
    if config.model_path.lora is not None:
        model = PeftModel.from_pretrained(model, config.model_path.lora, 
                                          torch_dtype=torch_dtype_dict[config.model_path.torch_dtype], #trust_remote_code=True,
                                         )
    print(model.hf_device_map)
    # bloom uses only fast tokenize
    to_use_fast = False
    if "bloom" in model_name_or_path or "mpt" in model_name_or_path:
        to_use_fast = True
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=to_use_fast) #, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None and "mpt" not in model_name_or_path:
        print("add special token")
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

    if 'chatglm' in model_name_or_path:
        labels="</s>"
        res = tokenizer.encode(labels, add_special_tokens=False)
        labels="<unk>"
        res = tokenizer.encode(labels, add_special_tokens=False)
        if len(res) > 1:
            tokens = ["</s>","<unk>"]
            tokenizer.add_tokens(tokens, special_tokens=True)
            labels="</s>"
            res = tokenizer.encode(labels, add_special_tokens=False)
        
        # print("")

    print("Show special token ids in tokenizer：",tokenizer.eos_token_id,tokenizer.pad_token_id, tokenizer.eos_token, tokenizer.pad_token)
    beam_size = config.generat.beam_size
    search = config.generat.search
    temperature = config.generat.temperature
    do_sample =  config.generat.do_sample
    if search == "sample":
        gen_config = GenerationConfig(temperature=temperature,
                                      top_p=0.9,
                                      do_sample=True,
                                      num_beams=beam_size,
                                      max_new_tokens=256,
                                      eos_token_id=tokenizer.eos_token_id,
                                      pad_token_id=tokenizer.pad_token_id,
                                      )
        

    elif search == "beam":
        gen_config = GenerationConfig(max_new_tokens=256,
                                      num_beams=beam_size,
                                      eos_token_id=tokenizer.eos_token_id,
                                      pad_token_id=tokenizer.pad_token_id,
                                      )

    else:
        raise ValueError("generat sample setting not right!")
        
        
    return model, tokenizer, gen_config



def load_generat_config(config, tokenizer):
    
    beam_size = config.generat.beam_size
    search = config.generat.search
    temperature = config.generat.temperature
    do_sample =  config.generat.do_sample
    if search == "sample":
        gen_config = GenerationConfig(temperature=temperature,
                                      top_p=0.9,
                                      do_sample=True,
                                      num_beams=beam_size,
                                      max_new_tokens=256,
                                      eos_token_id=tokenizer.eos_token_id,
                                      pad_token_id=tokenizer.pad_token_id,
                                      )
        

    elif search == "beam":
        gen_config = GenerationConfig(max_new_tokens=256,
                                      num_beams=beam_size,
                                      eos_token_id=tokenizer.eos_token_id,
                                      pad_token_id=tokenizer.pad_token_id,
                                      )

    else:
        raise ValueError("generat sample setting not right!")
        
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
    print(f"Load model...")
    model, tokenizer, gen_config = load_model(config)

    
    print(f"Begin generation...")
    input_data = data_dict["input_data"]
    prompt = data_dict["prompt_list"]
    input_file=data_dict["input_file"]
    print(f"Prepare output files...")
    
    record_base_path = config.model_path.lora if config.model_path.lora else config.model_path.base_model
    dataset_name = config.dataset.loader
    lang_pair_name = config.dataset.lang_pair if config.dataset.lang_pair else ""
    record_dir=os.path.join(record_base_path,dataset_name)
    os.makedirs(record_dir, exist_ok=True)
    if config.output.subpath is not None:
        record_dir=os.path.join(record_base_path,dataset_name, config.output.subpath)
        os.makedirs(record_dir, exist_ok=True)
    
    record_lang_dir=os.path.join(record_base_path,dataset_name,config.output.subpath,lang_pair_name)
    os.makedirs(record_lang_dir, exist_ok=True)
    lang_pair = config.dataset.lang_pair
    
    
    output_file = os.path.join(record_lang_dir, config.output.output_file_prefix+".txt")
    output_hyp_file = os.path.join(record_lang_dir, config.output.output_file_prefix+"_.hyp")
    # Generate
    torch.manual_seed(0)
    with open(output_file, 'w', encoding='utf-8') as fo, open(output_hyp_file, 'w', encoding='utf-8') as fo2:
        for i in tqdm(range(0, len(prompt), batch_size)):
            with torch.autocast("cuda"):
                p = prompt[i:i+batch_size]
                tokenized = tokenizer(p, padding=True, return_tensors="pt")
                input_ids = tokenized.input_ids.to(model.device)
                attn_mask = tokenized.attention_mask.to(model.device)
                input_ids = input_ids[:, :-1] if input_ids[0, -1] == tokenizer.eos_token_id else input_ids
                attn_mask = attn_mask[:, :-1] if input_ids[0, -1] == tokenizer.eos_token_id else attn_mask
                with torch.no_grad():
                    generated_ids = model.generate(inputs=input_ids, attention_mask=attn_mask, generation_config=gen_config,
                                                   pad_token_id=tokenizer.eos_token_id)


                decoded_tokens = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                for dec, gen_ids in zip(decoded_tokens, generated_ids):
                    print(dec, file=fo, flush=True)
                    print(post_process(dec), file=fo2, flush=True)
    hyps, refs, repeat_num = get_translation_from_hyp(output_file, os.path.dirname(input_file), lang_pair)
    score = get_spBLEU(hyps, refs)
    metric_file = os.path.join(os.path.dirname(output_file), "spBLEU_summary.csv")
    if os.path.exists(metric_file):
        with open(metric_file, 'a', encoding="utf-8") as writer:
            writer.writelines(f"{lang_pair} {score} \n")
    else:
        with open(metric_file, 'w', encoding="utf-8") as writer:
            writer.writelines(f"{lang_pair} {score} \n")
    print("Finished!")
    
def test_process_no_model_load(config, data_dict, model, tokenizer):
    batch_size = config.generat.batch_size
    gen_config = load_generat_config(config, tokenizer)
    
    print(f"Begin generation...")
    input_data = data_dict["input_data"]
    prompt = data_dict["prompt_list"]
    input_file=data_dict["input_file"]
    print(f"Prepare output files...")
    
    record_base_path = config.model_path.lora if config.model_path.lora else config.model_path.base_model
    dataset_name = config.dataset.loader
    lang_pair_name = config.dataset.lang_pair if config.dataset.lang_pair else ""
    record_dir=os.path.join(record_base_path,dataset_name)
    os.makedirs(record_dir, exist_ok=True)
    if config.output.subpath is not None:
        record_dir=os.path.join(record_base_path,dataset_name, config.output.subpath)
        os.makedirs(record_dir, exist_ok=True)
    
    record_lang_dir=os.path.join(record_base_path,dataset_name,config.output.subpath,lang_pair_name)
    os.makedirs(record_lang_dir, exist_ok=True)
    lang_pair = config.dataset.lang_pair
    
    
    output_file = os.path.join(record_lang_dir, config.output.output_file_prefix+".txt")
    output_hyp_file = os.path.join(record_lang_dir, config.output.output_file_prefix+"_.hyp")
    # Generate
    torch.manual_seed(0)
    with open(output_file, 'w', encoding='utf-8') as fo, open(output_hyp_file, 'w', encoding='utf-8') as fo2:
        for i in tqdm(range(0, len(prompt), batch_size)):
            p = prompt[i:i+batch_size]
            tokenized = tokenizer(p, padding=True, return_tensors="pt")
            input_ids = tokenized.input_ids.to(model.device)
            attn_mask = tokenized.attention_mask.to(model.device)
            input_ids = input_ids[:, :-1] if input_ids[0, -1] == tokenizer.eos_token_id else input_ids
            attn_mask = attn_mask[:, :-1] if input_ids[0, -1] == tokenizer.eos_token_id else attn_mask
            with torch.no_grad():
                generated_ids = model.generate(inputs=input_ids, attention_mask=attn_mask, generation_config=gen_config,
                                               pad_token_id=tokenizer.eos_token_id)


            decoded_tokens = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            for dec, gen_ids in zip(decoded_tokens, generated_ids):
                print(dec, file=fo, flush=True)
                print(post_process(dec), file=fo2, flush=True)
    hyps, refs, repeat_num = get_translation_from_hyp(output_file, os.path.dirname(input_file), lang_pair)
    score = get_spBLEU(hyps, refs)
    metric_file = os.path.join(os.path.dirname(output_file), "spBLEU_summary.csv")
    if os.path.exists(metric_file):
        with open(metric_file, 'a', encoding="utf-8") as writer:
            writer.writelines(f"{lang_pair} {score} \n")
    else:
        with open(metric_file, 'w', encoding="utf-8") as writer:
            writer.writelines(f"{lang_pair} {score} \n")
    print("Finished!")