import sys
from dataclasses import dataclass, field
from tqdm import tqdm
from typing import Optional, List
from datasets import load_dataset
import torch
import json
import transformers
import os
import re
import copy
import argparse
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, AutoConfig
import torch
from sklearn.metrics import accuracy_score
import pandas as pd
from peft import PeftModel
import sys
sys.path.append("/cpfs01/shared/NLP-A100/NLP-A100_hdd/jie/multilingual_LLM/scripts/trans.train/batch_train_chatglm/chatglm2-6b/")
from tokenization_chatglm import ChatGLMTokenizer
from modeling_chatglm import ChatGLMForConditionalGeneration
# from sklearn.metrics import accuracy_score
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "</s>"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


@dataclass
class GeneratingArguments:
    batch_size: int = field(default=8)
    output_file: str = field(default=None, metadata={"help": "Path to the output."})
    temperature: float = field(default=0.7)
    do_sample: bool = field(default=False)
    top_p: float = field(default=0.75)
    top_k: float = field(default=40)
    num_beams: int = field(default=1)
    max_new_tokens: int = field(default=512)
    template: str = field(default="alpaca")
    labels: Optional[List[str]] = field(default=None)
    evaluate: str = field(default="generate")
    
def load_model(config):
    # Load checkpoints
    torch_dtype_dict = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32
    }
    model_name_or_path = config.model_path.base_model
    if 'chatglm' in model_name_or_path:
        
        model = ChatGLMForConditionalGeneration.from_pretrained(
            model_name_or_path,
            load_in_8bit=config.model_path.load_in_8bit,
            torch_dtype=torch_dtype_dict[config.model_path.torch_dtype],trust_remote_code=True,
            device_map="auto",
        )
        if config.model_path.lora is not None:
            model = PeftModel.from_pretrained(model, config.model_path.lora, torch_dtype=torch_dtype_dict[config.model_path.torch_dtype],trust_remote_code=True)
        print(model.hf_device_map)
        model.eval()
        if torch.cuda.device_count() > 1:
            from accelerate import load_checkpoint_and_dispatch
            load_checkpoint_and_dispatch(
                model,
                model_name_or_path,
                device_map="auto",
                offload_state_dict=True,
                no_split_module_classes=["LlamaDecoderLayer"],
            )


        # bloom uses only fast tokenize
        to_use_fast = False
        if "bloom" in model_name_or_path or "mpt" in model_name_or_path:
            to_use_fast = True

        tokenizer = ChatGLMTokenizer.from_pretrained(config.model_path.base_model,  use_fast=to_use_fast, trust_remote_code=True)
    else:
        model_name_or_path = config.model_path.base_model
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            load_in_8bit=config.model_path.load_in_8bit,
            torch_dtype=torch_dtype_dict[config.model_path.torch_dtype],trust_remote_code=True,
            device_map="auto",
        )
        if config.model_path.lora is not None:
            model = PeftModel.from_pretrained(model, config.model_path.lora, torch_dtype=torch_dtype_dict[config.model_path.torch_dtype],trust_remote_code=True)
        print(model.hf_device_map)
        model.eval()
        if torch.cuda.device_count() > 1:
            from accelerate import load_checkpoint_and_dispatch
            load_checkpoint_and_dispatch(
                model,
                model_name_or_path,
                device_map="auto",
                offload_state_dict=True,
                no_split_module_classes=["LlamaDecoderLayer"],
            )


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
    if 'chatglm' in model_name_or_path:
        print(f"添加chatglm的特殊token，先查看现有的tokenizer是否支持：")
        labels="</s>"
        res = tokenizer.encode(labels, add_special_tokens=False)
        print("eos: ",res, len(res),"记录中的eos id为： ", tokenizer.eos_token_id)
        labels="<unk>"
        res = tokenizer.encode(labels, add_special_tokens=False)
        print("pad: ",res, len(res), "记录中的pad id为： ",tokenizer.pad_token_id)
        if len(res) > 1:
            print(f"现有的tokenizer不支持eos和pad，我们添加下")
            tokens = ["</s>","<unk>"]
            tokenizer.add_tokens(tokens, special_tokens=True)
            labels="</s>"
            res = tokenizer.encode(labels, add_special_tokens=False)
            print(f"添加eos和pad后，{res}")
    print("展示下，tokenizer的特殊符号id：",tokenizer.eos_token_id,tokenizer.pad_token_id, tokenizer.eos_token, tokenizer.pad_token)
        
    batch_size = config.generat.batch_size
    beam_size = config.generat.beam_size
    temperature = config.generat.temperature
    # output_file = config.output.output_file
    if config.dataset.labels is not None:
        force_words_ids = [tokenizer(["A", "B", "C", "D"], add_special_tokens=False).input_ids]
        print(f"采用了force word ids: {config.dataset.labels}, 对应的token id是{force_words_ids}")
    # gen_config = GenerationConfig(temperature=temperature,
    #                               top_p=0.9,
    #                               do_sample=False,
    #                               num_beams=beam_size,
    #                               max_new_tokens=1,
    #                               eos_token_id=tokenizer.eos_token_id,
    #                               pad_token=tokenizer.pad_token_id,
    #                               force_word_ids=[tokenizer(config.dataset.labels, add_special_tokens=False)[
    #                                                   "input_ids"]] if config.dataset.labels else None
    #                               # bad_words_ids=[tokenizer("\n", add_special_tokens=False)[
    #                               #                     "input_ids"]]
    #                               )
    # 上面是错的
    
    # gen_config = GenerationConfig(temperature=0.7,
    #                               top_p=0.9,
    #                               do_sample=False,
    #                               num_beams=beam_size,
    #                               max_new_tokens=1,
    #                               eos_token_id=tokenizer.eos_token_id,
    #                               pad_token=tokenizer.pad_token_id,
    #                               force_words_ids=force_words_ids if config.dataset.labels else None,
    #                               # force_word_ids=[tokenizer(config.dataset.labels, add_special_tokens=False)[
    #                               #                     "input_ids"]] if config.dataset.labels else None
    #                               # bad_words_ids=[tokenizer("\n", add_special_tokens=False)[
    #                               #                     "input_ids"]]
    #                               )
    # gen_config = GeneratingArguments
    if "mpt" in model_name_or_path:
        gen_config = GenerationConfig(
            temperature=config.generat.temperature,
            do_sample=config.generat.do_sample,
            top_p=config.generat.top_p,
            top_k=config.generat.top_k,
            num_beams=max(2, config.generat.num_beams) if config.dataset.labels else config.generat.num_beams,
            max_new_tokens=config.generat.max_new_tokens,
            force_words_ids=force_words_ids if config.dataset.labels else None,
            pad_token=tokenizer.pad_token_id,
            # force_word_ids=[tokenizer(config.dataset.labels, add_special_tokens=False)[
            #                     "input_ids"]] if config.dataset.labels else None
        )
    else:
        gen_config = GenerationConfig(
            temperature=config.generat.temperature,
            do_sample=config.generat.do_sample,
            top_p=config.generat.top_p,
            top_k=config.generat.top_k,
            num_beams=max(2, config.generat.num_beams) if config.dataset.labels else config.generat.num_beams,
            max_new_tokens=config.generat.max_new_tokens,
            force_words_ids=force_words_ids if config.dataset.labels else None,
            # force_word_ids=[tokenizer(config.dataset.labels, add_special_tokens=False)[
            #                     "input_ids"]] if config.dataset.labels else None
        )

    return model, tokenizer, gen_config

def load_generat_config(config, tokenizer):
    gen_config = GenerationConfig(
        temperature=config.generat.temperature,
        do_sample=config.generat.do_sample,
        top_p=config.generat.top_p,
        top_k=config.generat.top_k,
        num_beams=max(2, config.generat.num_beams) if config.dataset.labels else config.generat.num_beams,
        max_new_tokens=config.generat.max_new_tokens,
        force_words_ids=force_words_ids if config.dataset.labels else None,
    )
    return gen_config

def generate_prompt(instruction, input=None, template="alpaca"):
    if template == "alpaca":
        if input:
            return PROMPT_DICT["prompt_input"].format(instruction=instruction, input=input)
        else:
            return PROMPT_DICT["prompt_no_input"].format(instruction=instruction)
    elif template == "raw":
        if input:
            return f"{instruction}\n\n{input}"
        else:
            return f"{instruction}"
    else:
        raise NotImplementedError

def force_word_func(socre, hand_force_ids, tokenizer, num_beams):
    final_ids = []
    for max_score_id in torch.argmax(socre[:,hand_force_ids],dim=-1):
        final_ids.append(hand_force_ids[max_score_id])
    res=tokenizer.batch_decode(final_ids, skip_special_tokens=True)
    return res

# Post-process the output, extract translations
def post_process(text):
    text = text.split("### Response:")[1].strip()
    text = text.replace("\n", " ")
    # Cut for contrastive instruction
    if "</p>" in text:
        text = text.split("</p>")[0].split("<p>")[-1]
    return text

def evaluate_by_generate(
        model,
        tokenizer,
        dataset,
        template,
        generation_config,
        config
):
    prompt = [generate_prompt(ins, inp, template) for ins, inp in zip(dataset["instruction"], dataset["input"])]
    inputs = tokenizer(prompt, padding=True, return_tensors="pt").to("cuda")
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
        )
    output = tokenizer.batch_decode(generation_output.sequences, skip_special_tokens=True)
    # print(output)
    # print([tokenizer("A B C D", add_special_tokens=False)[
    #                         "input_ids"]])
    # print(tokenizer.decode(tokenizer("A B C D", add_special_tokens=False)[
    #                         "input_ids"]))
    # gen_config = GenerationConfig(
    #     do_sample=False,
    #     num_beams=1,
    #     max_new_tokens=1,
    #     force_word_ids=[tokenizer("A B C D", add_special_tokens=False)[
    #                         "input_ids"]]
    # )
    # with torch.no_grad():
    #     generation_output = model.generate(
    #         input_ids=inputs["input_ids"],
    #         attention_mask=inputs["attention_mask"],
    #         generation_config=gen_config,
    #         return_dict_in_generate=True,
    #         output_scores=True,
    #     )
    # print(generation_output.keys(),generation_output["scores"][-1].size())
    # print(inputs["input_ids"], generation_output.sequences)
    if config.generat.force_word_ids == 'oj':
        final_prediction = force_word_func(generation_output["scores"][-1], hand_force_ids=config.model_path.hand_force_ids, tokenizer=tokenizer, num_beams=config.generat.num_beams)
        return {**dataset, "prompt": prompt, "prediction": final_prediction}
    else:
        # return {**dataset, "prompt": prompt, "prediction": [o[len(p):].strip() for p, o in zip(prompt, output)]}
        return {**dataset, "prompt": prompt, "prediction": [post_process(o) for o in zip(output)]}
    # print(f"final prediction: {final_prediction}")
    # output = tokenizer.batch_decode(generation_output.sequences, skip_special_tokens=True)
    # print(output)
    # return {**dataset, "prompt": prompt, "prediction": [o[len(p):].strip() for p, o in zip(prompt, output)]}
    

def evaluate_by_perplexity(
        model,
        tokenizer,
        dataset,
        template,
        labels
):
    label_perplexity = []
    for label in labels:
        prompt = [generate_prompt(ins, inp, template) + label for ins, inp in
                  zip(dataset["instruction"], dataset["input"])]
        inputs = tokenizer(prompt, padding=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out_logits = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            ).logits
        shift_logits = out_logits[..., :-1, :].contiguous()
        shift_targets = inputs["input_ids"][..., 1:].contiguous()
        shift_attention_mask_batch = inputs["attention_mask"][..., 1:].contiguous()
        perplexity = torch.exp(
            (torch.nn.CrossEntropyLoss(reduction="none")(shift_logits.transpose(1, 2),
                                                         shift_targets) * shift_attention_mask_batch).sum(1)
            / shift_attention_mask_batch.sum(1)
        )
        label_perplexity.append(perplexity)
    prediction = [labels[l] for l in torch.stack(label_perplexity).argmin(dim=0).detach().cpu()]
    return {**dataset, "prediction": prediction}

def write_log(f, outputs):
    outputs = [dict(zip(outputs.keys(), t)) for t in zip(*outputs.values())]
    for output in outputs:
        f.write("\n")
        f.write("\n")
        f.write("#############################################################\n")
        # print(output)
        # print(type(output))
        for k, v in output.items():
            if k=="prompt":
                f.write("--------------------------------------------------\n")
            f.write(f"{k}: {v}")
            f.write("\n")
            if k=="prompt":
                f.write("--------------------------------------------------\n")

def test_process(config, data_dict):
    PROMPT_DICT = data_dict["PROMPT_DICT"]
    test_dataset = data_dict["test_dataset"]
    batch_size = config.generat.batch_size
    print(f"加载模型...")
    model, tokenizer, gen_config = load_model(config)
    print(f"模型加载完成")
    

    config.dataset.lang_pair = config.dataset.path.split('/')[-1].split('_')[-2]
    print(f"准备结果记录日志...")
    record_base_path = config.model_path.lora if config.model_path.lora else config.model_path.base_model
    dataset_name = config.dataset.loader
    lang_pair_name = config.dataset.lang_pair if config.dataset.lang_pair else ""
    record_dir=os.path.join(record_base_path,dataset_name)
    os.makedirs(record_dir, exist_ok=True)
    record_lang_dir=os.path.join(record_base_path,dataset_name,lang_pair_name)
    os.makedirs(record_lang_dir, exist_ok=True)
    
    # output_file = os.path.join(record_lang_dir, config.output.output_file_prefix+".txt")
    output_raw_file = os.path.join(record_lang_dir, config.output.output_file_prefix+"_raw.txt")
    
    print(f"开始生成...")
    right_sample = 0.
    total_sample = 0.
    total_gt = []
    total_pred = []
    with open(output_raw_file, "w") as output_file:
        for i in tqdm(range(0, len(test_dataset), batch_size)):
            d = test_dataset[i:i + batch_size]
            if config.metric.type == "generate":
                output = evaluate_by_generate(model, tokenizer, d, template=config.generat.template, generation_config=gen_config, config=config)
            elif config.metric.type == "perplexity":
                assert config.dataset.labels, "evaluate with perplexity requires labels"
                output = evaluate_by_perplexity(model, tokenizer, d, template=config.generat.template, labels=config.dataset.labels)
            # 写入日志
            # sys.exit(0)
            write_log(output_file, output)
            for gt, pred in zip(output["output"], output["prediction"]):
                total_gt.append(gt)
                total_pred.append(pred.strip())
                if gt == pred.strip():
                    right_sample +=1
                total_sample +=1
            
            output_file.flush()
        accuracy = accuracy_score(total_gt, total_pred)
        print(f"cur acc: {right_sample/total_sample}")
        print(f"api cur acc: {accuracy*100}")
        output_file.writelines(
            json.dumps({"accuracy": accuracy*100}, ensure_ascii=False) + "\n" 
        )
        
def test_process_no_model_load(config, data_dict, model, tokenizer):
    PROMPT_DICT = data_dict["PROMPT_DICT"]
    test_dataset = data_dict["test_dataset"]
    batch_size = config.generat.batch_size
    # print(f"加载模型...")
    # model, tokenizer, gen_config = load_model(config)
    # print(f"模型加载完成")
    gen_config = load_generat_config(config, tokenizer)
    

    config.dataset.lang_pair = config.dataset.path.split('/')[-1].split('_')[-2]
    print(f"准备结果记录日志...")
    record_base_path = config.model_path.lora if config.model_path.lora else config.model_path.base_model
    dataset_name = config.dataset.loader
    lang_pair_name = config.dataset.lang_pair if config.dataset.lang_pair else ""
    record_dir=os.path.join(record_base_path,dataset_name)
    os.makedirs(record_dir, exist_ok=True)
    record_lang_dir=os.path.join(record_base_path,dataset_name,lang_pair_name)
    os.makedirs(record_lang_dir, exist_ok=True)
    
    # output_file = os.path.join(record_lang_dir, config.output.output_file_prefix+".txt")
    output_raw_file = os.path.join(record_lang_dir, config.output.output_file_prefix+"_raw.txt")
    
    print(f"开始生成...")
    right_sample = 0.
    total_sample = 0.
    total_gt = []
    total_pred = []
    with open(output_raw_file, "w") as output_file:
        for i in tqdm(range(0, len(test_dataset), batch_size)):
            d = test_dataset[i:i + batch_size]
            if config.metric.type == "generate":
                output = evaluate_by_generate(model, tokenizer, d, template=config.generat.template, generation_config=gen_config, config=config)
            elif config.metric.type == "perplexity":
                assert config.dataset.labels, "evaluate with perplexity requires labels"
                output = evaluate_by_perplexity(model, tokenizer, d, template=config.generat.template, labels=config.dataset.labels)
            # 写入日志
            # sys.exit(0)
            write_log(output_file, output)
            for gt, pred in zip(output["output"], output["prediction"]):
                total_gt.append(gt)
                total_pred.append(pred.strip())
                if gt == pred.strip():
                    right_sample +=1
                total_sample +=1
            
            output_file.flush()
        accuracy = accuracy_score(total_gt, total_pred)
        print(f"cur acc: {right_sample/total_sample}")
        print(f"api cur acc: {accuracy*100}")
        output_file.writelines(
            json.dumps({"accuracy": accuracy*100}, ensure_ascii=False) + "\n" 
        )