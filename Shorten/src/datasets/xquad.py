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
import json

# Special tokens in llama
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "</s>"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"),
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
                instruct = instruct.replace('['+key+']', val)

            inst_list.append(instruct)

    return inst_list

# Read input data for inference
def read_input(path, size):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)['data']
        input_data = []
        for paragraphs_title in data:
            paragraphs = paragraphs_title['paragraphs']
            for context_QAs in paragraphs:
                context = context_QAs['context']
                QAs = context_QAs['qas']
                for answers_id_question in QAs:
                    answer = answers_id_question['answers'][0]['text']
                    id = answers_id_question['id']
                    question = answers_id_question['question']
                    input_data.append({'context': context,
                                       'answer': answer,
                                       'question': question,
                                       'id': id})
        
    return input_data[:size]

# Assembly instruction and input data, handle hints
def prepare_prompt(instruction_list, input_data=None, input_extra_data=None, inst_with_input=False):
    prompt_dict = [{"instruction": instruct, 
                    "input": input['context'] + ' ' + input['question']} \
                    for instruct, input in zip(instruction_list, input_data)]

    prompt_input = PROMPT_DICT['prompt_input']
    sources = [prompt_input.format_map(example) for example in prompt_dict]

    return sources


def dataloader(config):

    inst_fix = config.dataset.inst_fix
    inst_index = config.dataset.inst_index if inst_fix else None
    inst_with_input = config.dataset.inst_with_input
    inst_placeholder_list = config.dataset.inst_placeholder


    if inst_placeholder_list is None:
        inst_placeholder = {}
    else:
        inst_placeholder = {inst_placeholder_list[i]: \
                            inst_placeholder_list[i+1] \
                            for i in range(0, len(inst_placeholder_list), 2)}
    
    
    input_file = os.path.join(config.dataset.path, config.dataset.input_file+f".{config.dataset.lang_pair}.json")
    input_size = config.dataset.input_size
    input_extra_file = os.path.join(config.dataset.path, config.dataset.input_extra_file) if config.dataset.input_extra_file is not None else None 
    inst_file  = os.path.join(config.dataset.path, config.dataset.inst_file)
    
    if input_file is not None:
        input_data = read_input(input_file, input_size)
    else:
        input_data = None
    
    if input_extra_file is not None:
        input_extra_data = read_input(input_extra_file)
    else:
        input_extra_data = None

    # Prepare input data
    if inst_file is not None:
        instruction_list = read_instruct(inst_file, inst_placeholder)
    else: # In case instruction file is missing, then use input as instruction
        instruction_list = []

    if inst_fix:
        instruction_list = [instruction_list[inst_index]]*len(input_data)

    id_list = [data['id'] for data in input_data]
    prompt_list = prepare_prompt(instruction_list=instruction_list, 
                                input_data=input_data, 
                                input_extra_data=input_extra_data, 
                                inst_with_input=inst_with_input)
    
    data_dict = {
        "prompt_list": prompt_list,
        "input_data": input_data,
        "id_list": id_list
    }
    return data_dict
