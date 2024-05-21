import sys
from dataclasses import dataclass, field
from tqdm import tqdm
from typing import Optional, List
from datasets import load_dataset
import torch
import json
import transformers
from transformers import GenerationConfig
import os
import re
import copy

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "<pad>"
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

# def smart_tokenizer_and_embedding_resize(
#     special_tokens_dict: Dict,
#     tokenizer: transformers.PreTrainedTokenizer,
#     model: transformers.PreTrainedModel,
# ):
#     """Resize tokenizer and embedding.

#     Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
#     """
#     num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
#     model.resize_token_embeddings(len(tokenizer))

#     if num_new_tokens > 0:
#         input_embeddings = model.get_input_embeddings().weight.data
#         output_embeddings = model.get_output_embeddings().weight.data

#         input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
#         output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

#         input_embeddings[-num_new_tokens:] = input_embeddings_avg
#         output_embeddings[-num_new_tokens:] = output_embeddings_avg

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


def dataloader(config):
    data_path_base, data_path_name = config.dataset.path.rsplit(os.path.sep, maxsplit=1)
    dataset_name, dataset_config = data_path_name.split("_", maxsplit=1)
    print(data_path_base, dataset_name,dataset_config)
    test_dataset = load_dataset(os.path.join(data_path_base, dataset_name), config=dataset_config, split="test")
    data_dict={"test_dataset": test_dataset,
              "PROMPT_DICT": PROMPT_DICT}
    return data_dict
