import argparse
import logging
import sys
# from src.datasets.common import get_lang_instruction
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, AutoConfig
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import os
import pandas as pd
from peft import PeftModel

# Special tokens in llama
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "</s>" # 这里要用</s>, [PAD]
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
def read_input(path):
    df = pd.read_csv(path)
    return df


# Assembly instruction and input data, handle hints
def prepare_prompt(instruction_list, input_data=None, input_extra_data=None, inst_with_input=False):
    prompt_dict = [{"instruction": instruct,
                    "input": f"Sentence 1 | {input[0]} Sentence 2 | {input[1]} "} \
                   for instruct, input in zip(instruction_list, input_data)]

    prompt_input = PROMPT_DICT['prompt_input']
    sources = [prompt_input.format_map(example) for example in prompt_dict]

    return sources





def dataloader(config):


    input_file = os.path.join(config.dataset.path, config.dataset.lang_pair, config.dataset.input_file)
    
    
    # print(type(config.dataset.input_extra_file))
    if input_file is not None:
        input_data = read_input(input_file)
    else:
        input_data = None

    data_dict = {
        "input_data": input_data
    }
    return data_dict
