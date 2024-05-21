import argparse
import logging
import sys
from src.datasets.common import get_lang_instruction
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, AutoConfig
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import os
import pandas as pd
from peft import PeftModel

# Special tokens in llama
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "</s>"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is a series of different language sentences. You will be given a question at the end, after the examples, for you to answer."
        "### Input:\n{input}\nChoices:\n(A) entailment\n(B) neutral\n(C) contradiction\n Which one of the three choices is correct relationship for the two sentences, (A), (B) or  (C) \n### Response:"),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n"
        "### Instruction:\n{instruction}\n### Response:"
    ),
}


# Read task instruction, fill in languages
def read_instruct(inst_file, inst_placeholder={}):
    inst_list = []
    with open(inst_file, 'r', encoding='utf-8') as f:
        instructions = f.readlines()
        for instruct in instructions:
            instruct = instruct.strip()
            for key, val in inst_placeholder.items():
                # replace [key] with val
                instruct = instruct.replace('[' + key + ']', val)

            inst_list.append(instruct)

    return inst_list


# Read input data for inference
def read_input(path, size, lang_pair, use_match=False):
    label_map = {"entailment": "A",
             "neutral":"B",
            "contradiction":"C"}
    df = pd.read_csv(path, sep="\t")[["language", "gold_label", "sentence1", "sentence2", "match"]]
    df_lang = df.loc[df['language'] == lang_pair]
    df_lang = df_lang.head(size) if size != -1 else df_lang
    if use_match:
        df_lang = df_lang.loc[df_lang['match'] == True]
    df_lang['target'] = df_lang['gold_label'].map(label_map)
    return df_lang.values.tolist()
    
#     src, tgt = lang_pair.split("-")

#     df = pd.read_csv(path, sep="\t")[[src, tgt]]

#     df = df.head(size) if size != -1 else df

#     return df.values.tolist()


# Assembly instruction and input data, handle hints
def prepare_prompt(instruction_list, input_data=None, input_extra_data=None, inst_with_input=False):
    prompt_dict = [{"instruction": instruct,
                    "input": f"Sentence 1 | {input[2]} Sentence 2 | {input[3]} "} \
                   for instruct, input in zip(instruction_list, input_data)]

    prompt_input = PROMPT_DICT['prompt_input']
    sources = [prompt_input.format_map(example) for example in prompt_dict]

    return sources





def dataloader(config):

    inst_fix = config.dataset.inst_fix
    inst_index = config.dataset.inst_index if inst_fix else None
    inst_with_input = config.dataset.inst_with_input
    inst_placeholder_list = config.dataset.inst_placeholder

    lang_instruction = get_lang_instruction()
    # print(lang_instruction)
    # src, tgt = config.dataset.lang_pair.split("-")
    # source, target = lang_instruction[src], lang_instruction[tgt]
    
    if inst_placeholder_list is None:
        inst_placeholder = {"SRC": config.dataset.lang_pair, "TGT": config.dataset.lang_pair}
    else:
        inst_placeholder = {inst_placeholder_list[i]: \
                                inst_placeholder_list[i + 1] \
                            for i in range(0, len(inst_placeholder_list), 2)}
    
    
    input_file = os.path.join(config.dataset.path, config.dataset.input_file)
    input_size = config.dataset.input_size
    input_extra_file = os.path.join(config.dataset.path, config.dataset.input_extra_file) if config.dataset.input_extra_file is not None else None 
    inst_file  = os.path.join(config.dataset.path, config.dataset.inst_file)
    
    # print(type(config.dataset.input_extra_file))
    if input_file is not None:
        input_data = read_input(input_file, input_size, config.dataset.lang_pair, config.dataset.use_match)
    else:
        input_data = None
    # print(input_data)
    
    if input_extra_file is not None:
        input_extra_data = read_input(input_extra_file)
    else:
        input_extra_data = None

    # Prepare input data
    if inst_file is not None:
        instruction_list = read_instruct(inst_file, inst_placeholder)
    else:  # In case instruction file is missing, then use input as instruction
        instruction_list = []

    if inst_fix:
        instruction_list = [instruction_list[inst_index]] * len(input_data)
    # print(instruction_list)
    prompt_list = prepare_prompt(instruction_list=instruction_list,
                                 input_data=input_data,
                                 input_extra_data=input_extra_data,
                                 inst_with_input=inst_with_input)
    
    data_dict = {
        "prompt_list": prompt_list,
        "input_data": input_data
    }
    return data_dict
