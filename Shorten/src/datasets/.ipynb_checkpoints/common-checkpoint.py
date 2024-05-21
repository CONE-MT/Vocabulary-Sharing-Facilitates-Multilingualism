from collections import Counter

import pandas as pd
import torch
import os
import sacrebleu
from transformers import AutoTokenizer, LlamaTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 20

import seaborn as sns


top20_lgs = "ro es de ca ceb pt da gl oc no  jv luo ms ga ns so hr bs ln".split()
llama_lgs = "bg ca cs da de es fr hr hu it nl pl pt ro ru sl sr sv uk".split()

# top20_lgs = "ro es de ca ceb pt gl oc no  jv luo ms ga ns so hr bs".split()
# top20_lgs = "ln da".split()
# top20_lgs = "ro es de ca  pt da".split()

LGS_DICT = {"afr": "af", "amh": "am", "ara": "ar", "hye": "hy", "asm": "as", "ast": "ast", "azj": 'az', "bel": 'be',
            "ben": 'bn', "bos": 'bs', "bul": 'bg', "mya": 'my',
            "cat": 'ca', "ceb": 'ceb', "zho_simpl": 'zh', "zho_trad": 'zhtrad', "hrv": 'hr', "ces": 'cs', "dan": 'da',
            "nld": 'nl', "eng": 'en', "est": 'et', "tgl": 'tl',
            "fin": 'fi', "fra": 'fr', "ful": 'ff', "glg": 'gl', "lug": 'lg', "kat": 'ka', "deu": 'de', "ell": 'el',
            "guj": 'gu', "hau": 'ha', "heb": 'he', "hin": 'hi', "Latvian": "lij",
            "hun": 'hu', "isl": 'is', "ibo": 'ig', "ind": 'id', "gle": 'ga', "ita": 'it', "jpn": 'ja', "jav": 'jv',
            "kea": 'kea', "kam": 'kam', "kan": 'kn', "kaz": 'kk',
            "khm": 'km', "kor": 'ko', "kir": 'ky', "lao": 'lo', "lav": 'lv', "lin": 'ln', "lit": 'lt', "luo": 'luo',
            "ltz": 'lb', "mkd": 'mk', "msa": 'ms', "mal": 'ml',
            "mlt": 'mt', "mri": 'mi', "mar": 'mr', "mon": 'mn', "npi": 'ne', "nso": 'ns', "nob": 'no', "nya": 'ny',
            "oci": 'oc', "ory": 'or', "orm": 'om', "pus": 'ps',
            "fas": 'fa', "pol": 'pl', "por": 'pt', "pan": 'pa', "ron": 'ro', "rus": 'ru', "srp": 'sr', "sna": 'sn',
            "snd": 'sd', "slk": 'sk', "slv": 'sl', "som": 'so',
            "ckb": 'ku', "spa": 'es', "swh": 'sw', "swe": 'sv', "tgk": 'tg', "tam": 'ta', "tel": 'te', "tha": 'th',
            "tur": 'tr', "ukr": 'uk', "umb": 'umb', "urd": 'ur',
            "uzb": 'uz', "vie": 'vi', "cym": 'cy', "wol": 'wo', "xho": 'xh', "yor": 'yo', "zul": 'zu'}

REVERSE_LGS_DICT = {v: k for k, v in LGS_DICT.items()}

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

def plot_two_axes(res_df, x, y1, y2, y1_label, y2_label, png_file=None):
    fig = plt.figure(figsize=(13, 7))
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()

    sns.barplot(x=x, y=y1, data=res_df, ax=ax1, color="#d3acac")
    sns.lineplot(x=x, y=y2, data=res_df, ax=ax2, color="#242124")

    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)

    plt.legend([y1_label, y2_label], loc="best", fontsize=20)

    plt.tight_layout()
    ax1.grid(False)
    ax2.grid(False)
    if png_file is None:
        plt.show()
    else:
        plt.savefig(png_file)


def plot_png_with_line(res_df, x, y1, y1_label, png_file, hue):
    sns.lineplot(x=x, y=y1, data=res_df, hue=hue)

    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    plt.legend(ncol=5, loc="best")

    plt.tight_layout()
    plt.show()
    # plt.savefig(png_file)


def plot_png_with_bar(res_df, x, y1, hue, png_file=None, title=None):
    sns.barplot(x=x, y=y1, data=res_df, hue=hue)

    plt.yticks(fontsize=20)
    plt.xticks(fontsize=15, rotation=70)
    plt.ylim(res_df[y1].min() * 0.9, res_df[y1].max() * 1.1)
    plt.legend(ncol=5, loc="best", fontsize=10)
    if title is not None:
        plt.title(title)
    plt.tight_layout()

    if png_file is None:
        plt.show()
    else:
        plt.savefig(png_file)



def get_lang_instruction():
    df = pd.read_excel(
        "/cpfs01/shared/NLP-A100/NLP-A100_hdd/jie/multilingual_LLM/prepare/language_name_code_pair.xlsx")
    return dict(zip(df['language_code'], df['language_en']))


def get_embedding_tensor_and_tokenizer(model_path):
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
    model.eval()
    input_embeddings = model.get_input_embeddings()
    return input_embeddings, tokenizer


def get_translation_from_hyp(hyp_file, ref_dir, lang_pair):
    format_string = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request."
    src, trg = lang_pair.split("-")
    src_file_name = os.path.join(ref_dir, f"{REVERSE_LGS_DICT[src]}.devtest")
    trg_file_name = os.path.join(ref_dir, f"{REVERSE_LGS_DICT[trg]}.devtest")
    print(f"src_file_name:{src_file_name}, trg_file_name:{trg_file_name}")
    hyps_string, hyps, copy_ratio = "", [], []
    with open(hyp_file, 'r', encoding="utf-8") as reader, \
            open(src_file_name, 'r', encoding="utf-8") as r_reader, \
            open(trg_file_name, 'r', encoding="utf-8") as t_reader:
        for line in reader:
            hyps_string += line.strip()

        refs = [l.strip() for l in t_reader.readlines()]
        src_inputs = [l.strip() for l in r_reader.readlines()]

    parts = hyps_string.split(format_string)
    for part in parts:
        if len(part) == 0:
            continue
        tmp_ps = part.split("###")
        tmp_ref = src_inputs[len(hyps)].strip()
        hyp = ""
        for i, tmp in enumerate(tmp_ps):
            if tmp.startswith(" Input") and tmp != " Input":
                t_input = tmp.split(":")[1].strip()
                if tmp_ref.startswith(t_input) or sacrebleu.sentence_bleu(t_input, [tmp_ref],
                                                                          tokenize="spm").score > 0.9:
                    if i + 1 >= len(tmp_ps):
                        break
                    else:
                        x = tmp_ps[i + 1].split(":")
                        hyp = x[1].strip() if len(x) > 1 else ""
                        if len(hyp) > 0:
                            break
        tmp_counter = Counter(hyp.split())
        tmp = [k for k, v in tmp_counter.items() if v > 2]
        ratio = len(tmp) / len(tmp_counter) if len(hyp) != 0 else 0
        copy_ratio.append(ratio)

        hyps.append(hyp)
        if len(hyps) == len(refs):
            break

    repeat_num = len([i for i in copy_ratio if i > 0.5])
    return hyps, refs, repeat_num


def get_spBLEU(hyps, refs):
    if len(hyps) != len(refs):
        return None
    result = sacrebleu.corpus_bleu(hyps, [refs], tokenize="spm", force=True).score
    return result