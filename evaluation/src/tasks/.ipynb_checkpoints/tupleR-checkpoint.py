import argparse
import logging
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, AutoConfig
import torch
from tqdm import tqdm
# from sklearn.metrics import accuracy_score
import os
import pandas as pd
from peft import PeftModel



def load_model(config):
    torch_dtype_dict = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32
    }
    
    # Load checkpoints
    model_name_or_path = config.model_path.base_model
    model = AutoModelForCausalLM.from_pretrained(config.model_path.base_model, torch_dtype=torch_dtype_dict[config.model_path.torch_dtype], device_map="auto")
    if config.model_path.lora is not None:
        model = PeftModel.from_pretrained(model, config.model_path.lora, torch_dtype=torch_dtype_dict[config.model_path.torch_dtype])
    print(model.hf_device_map)
    # bloom uses only fast tokenize
    to_use_fast = False
    if "bloom" in model_name_or_path:
        to_use_fast = True

    tokenizer = AutoTokenizer.from_pretrained(config.model_path.base_model,  use_fast=to_use_fast)
    tokenizer.padding_side = "left"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    batch_size = config.generat.batch_size
    beam_size = config.generat.beam_size
    temperature = config.generat.temperature
    # output_file = config.output.output_file

    gen_config = GenerationConfig(temperature=temperature,
                                  top_p=0.9,
                                  do_sample=False,
                                  num_beams=1,
                                  max_new_tokens=config.generat.max_new_tokens,
                                  eos_token_id=tokenizer.eos_token_id,
                                  pad_token=tokenizer.pad_token_id,
                                  # force_word_ids=[tokenizer(config.dataset.labels, add_special_tokens=False)[
                                  #                     "input_ids"]] if config.dataset.labels else None
                                  )
    
    return model, tokenizer, gen_config

def post_process_3(pred_text, src_text, target_text):
    # print(src_text)
    input_text = src_text.split("### Input:\n")[1]
    input_text = input_text.split("___")[0]
    # print(f"input_text: {input_text}")
    prediction_text = pred_text.split(src_text)[-1]
    # print(f"prediction_text: {prediction_text}")
    pred_result = prediction_text.split(input_text)[-1][:len(target_text)]
    # print(f"pred_result: {pred_result}| target_text: {target_text}")
    return pred_result

def post_process_1(pred_text, src_text, target_text):
    return pred_text.split(src_text)[-1][:len(target_text)]

def accuracy_score(gts, preds):
    right_count = 0.
    all_count = 0.
    for gt, pred in zip(gts, preds):
        if gt.lower()==pred.lower():
            right_count+=1
        all_count+=1
    return right_count/all_count

# Post-process the output, extract translations
def post_process(pred_text, src_text, target_text):
    return pred_text.split(src_text)[-1][1:len(target_text)+1]


POST_PROCESS = {
    "post_process_clm": post_process,
    "post_process_3": post_process_3,
    "post_process_1": post_process_1
}

def test_process(config, data_dict):
    batch_size = config.generat.batch_size
    print(f"加载模型...")
    model, tokenizer, gen_config = load_model(config)
    print(f"模型加载完成")
    
    print(f"开始生成...")
    input_data = data_dict["input_data"]
    # prompt_list = data_dict["prompt_list"]
    print(f"准备结果记录日志...")
    
    record_base_path = config.model_path.lora if config.model_path.lora else config.model_path.base_model
    dataset_name = config.dataset.loader
    lang_pair_name = config.dataset.lang_pair if config.dataset.lang_pair else ""
    record_dir=os.path.join(record_base_path,dataset_name)
    os.makedirs(record_dir, exist_ok=True)
    record_lang_dir=os.path.join(record_base_path,dataset_name,lang_pair_name)
    if config.output.subpath is not None:
        record_lang_dir=os.path.join(record_lang_dir, config.output.subpath)
    os.makedirs(record_lang_dir, exist_ok=True)
    
    output_file = os.path.join(record_lang_dir, config.output.output_file_prefix+".txt")
    output_raw_file = os.path.join(record_lang_dir, config.output.output_file_prefix+"_raw.txt")
    output_csv_file = os.path.join(record_lang_dir, config.output.output_file_prefix+".csv")
    # 获取表头
    header = input_data.columns.tolist()
    print("Header:", header)
    # 初始化一个空列表来保存新的数据
    target_columns = config.generat.target_column
    prefix_columns = config.generat.prefix_column
    print(f"使用的数据列为: {target_columns}, {prefix_columns}")
    
    new_data = []
    # Generate
    # targets = #[data[-1] for data in input_data]
    predictions = []
    # path_split = output_file.split('.')
    # output_raw_file = '.'.join(path_split[:-1] + ['raw'] + ['txt'])
    # os.makedirs(os.path.dirname(output_file), exist_ok=True)
    print(f"样本个数：{len(input_data)}")
    with open(output_file, 'w', encoding='utf-8') as f, open(output_raw_file, 'w') as g:
        # for i in tqdm(range(0, 50, batch_size)):
        for i in tqdm(range(0, len(input_data), batch_size)):
            p = input_data[i:i + batch_size][prefix_columns].tolist()
            row_data = input_data[i:i + batch_size]
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
            # print(tokenizer.batch_decode(generated_ids, skip_special_tokens=False))
            # print(decoded_tokens)
            targets = input_data[i:i + batch_size][target_columns].tolist()
            targets = [x.split("</s>")[0] for x in targets]
            sample_ids = input_data[i:i + batch_size]["sample_id"].tolist()
            pattern_ids = input_data[i:i + batch_size]["pattern_id"].tolist()
            entities1 =  input_data[i:i + batch_size]["entity1"].tolist()
            entities2 =  input_data[i:i + batch_size]["entity2"].tolist()
            patterns =  input_data[i:i + batch_size]["pattern"].tolist()
            for src_text, pred_text, row, target_text in zip(p, decoded_tokens, row_data.values, targets):
                prediction = POST_PROCESS[config.generat.postprocess](pred_text, src_text, target_text)
                determine = True if prediction==target_text else False
                new_row = row.tolist() + [prediction, determine]
                new_data.append(new_row)
                
            
            for pred_text, src_text, target_text, s_id, p_id, e1, e2, pat in zip(decoded_tokens, p, targets, sample_ids, pattern_ids, entities1, entities2, patterns):
                # print(dec)
                # print("-------------------------------------")
                prediction = POST_PROCESS[config.generat.postprocess](pred_text, src_text, target_text)
                predictions.append(prediction)
                # print("--------------------------------------------")
                # print(f"pred_text: {pred_text}")
                # print(f"pred: {prediction}, target: {target_text}")
                
                f.write(prediction)
                f.write('\n')
                
                
                g.write("--------------------------------------------")
                g.write('\n')
                g.write(f"sample id: {s_id}, pattern_id: {p_id}")
                g.write('\n')
                g.write(f"entity1: {e1}, entity2: {e2}, pattern: {pat}")
                g.write('\n')
                g.write("****************************************************")
                g.write('\n')
                g.write(f"input_text: {src_text}")
                g.write('\n')
                g.write("****************************************************")
                g.write('\n')
                g.write(f"pred_text: {pred_text}")
                g.write('\n')
                g.write("****************************************************")
                g.write('\n')
                g.write(f"pred: {prediction}, target: {target_text}")
                
                g.write('\n')
                g.write('\n')
                
        targets = input_data[:len(predictions)][target_columns].tolist()
        targets = [x.split("</s>")[0] for x in targets]
        accuracy = accuracy_score(targets, predictions)
        accuracy_info = 'accuracy: {}'.format(float(accuracy))
        print(f"accuracy: {accuracy}")
        f.write(accuracy_info)
        # 创建一个新的 DataFrame
        new_df = pd.DataFrame(new_data, columns=input_data.columns.tolist() + ['prediction', 'determine'])
        # 保存新的 DataFrame 到 CSV 文件
        new_df.to_csv(output_csv_file, index=False)
    print("Finished!")