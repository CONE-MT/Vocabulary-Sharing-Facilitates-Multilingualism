import os
import datetime
import sys
import time
import json
import fnmatch
import pandas as pd
from subprocess import call
from datetime import datetime, timedelta
import datetime as dt
import argparse

sys.path.append("/cpfs01/shared/NLP-A100/NLP-A100_hdd/jie/multilingual_LLM/")
from eval import read_yaml_file, AttributeDict

SUPPORT_XNLI = ["ar", "bg", "de", "el", "en", "es", "fr", "hi", "ru", "sw", "th", "tr", "ur", "vi", "zh"]
SUPPORT_XQUAD = ["ar", "de", "el", "en", "es", "hi", "ro", "ru", "th", "tr", "vi", "zh"]
SUPPORT_MLQA=["en",	"de","es","ar",	"zh", "vi", "hi"]
SUPPORT_MMLU_FLORES = ["af", "am", "ar", "as", "ast", "az", "be", "bg", "bn", "bs", "ca", "ceb", "cs", "cy", "da", "de", "el", "en", "es", "et", "fa", "ff", "fi", "fr", "ga", "gl", "gu", "ha", "he", "hi", "hr", "hu", "hy", "id", "ig", "is", "it", "ja", "jv", "ka", "kam", "kea", "kk", "km", "kn", "ko", "ku", "ky", "lb", "lg", "ln", "lo", "lt", "luo", "lv", "mi", "mk", "ml", "mn", "mr", "ms", "mt", "my", "ne", "nl", "no", "ns", "ny", "oc", "om", "or", "pa", "pl", "ps", "pt", "ro", "ru", "sd", "sk", "sl", "sn", "so", "sr", "sv", "sw", "ta", "te", "tg", "th", "tl", "tr", "uk", "umb", "ur", "uz", "vi", "wo", "xh", "yo", "zh", "zu"]

BASE_SHELL = '''
bash {DLC_CONFIG_PATH}\n
echo {COMMAND}\n
{CPFS_PATH}/dlc create job \
    --command "{COMMAND}" \
    --data_sources "" \
    --kind "PyTorchJob" \
    --name "{NAME}" \
    --node_names "" \
    --priority "4" \
    --resource_id "quotactf01wuemor" \
    --thirdparty_lib_dir "" \
    --thirdparty_libs "" \
    --worker_count "1" \
    --worker_cpu "4" \
    --worker_gpu "1" \
    --worker_gpu_type "nvidia_a100-sxm4-80gb" \
    --worker_image "pjlab-wulan-acr-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/pjlab-eflops/yuanshuai:llama-jie-v1" \
    --worker_memory "32Gi" \
    --worker_shared_memory "32Gi" \
    --workspace_id "wsvnint04sctwu5a"
'''
BASE_MLQA_SHELL = '''
bash {DLC_CONFIG_PATH}\n
echo {COMMAND}\n
{CPFS_PATH}/dlc create job \
    --command "{COMMAND}" \
    --data_sources "" \
    --kind "PyTorchJob" \
    --name "{NAME}" \
    --node_names "" \
    --priority "2" \
    --resource_id "quotactf01wuemor" \
    --thirdparty_lib_dir "" \
    --thirdparty_libs "" \
    --worker_count "1" \
    --worker_cpu "4" \
    --worker_gpu "1" \
    --worker_gpu_type "nvidia_a100-sxm4-80gb" \
    --worker_image "pjlab-wulan-acr-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/pjlab-eflops/yuanshuai:llama-jie-v1" \
    --worker_memory "32Gi" \
    --worker_shared_memory "32Gi" \
    --workspace_id "wsvnint04sctwu5a"
'''
CPFS_PATH="/cpfs01/user/yuanshuai/"
DLC_CONFIG_PATH = "/cpfs01/shared/NLP-A100/NLP-A100_hdd/jie/multilingual_LLM/src/utils/dlc.sh"
PYTHON_PATH = "/cpfs01/shared/NLP-A100/NLP-A100_hdd/jie/anaconda3/envs/llm2/bin/python"
EVAL_SCRIPT_PATH = "/cpfs01/shared/NLP-A100/NLP-A100_hdd/jie/multilingual_LLM/eval.py"
BASE_CONFIGS_PATH = "/cpfs01/shared/NLP-A100/NLP-A100_hdd/jie/multilingual_LLM/configs/A100/"
PROJECT_PATH="/cpfs01/shared/NLP-A100/NLP-A100_hdd/jie/multilingual_LLM/"
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
COPY_CONFIG_DICT={
    "xnli": ["generation_results_raw.txt", "generation_results.txt"], # generation_results.txt 最后一行
    "mmlu": ["generation_results_raw.txt"], # 最后一行 {"accuracy": 0.0}
    "mlqa": ["generation_results_raw.txt", "generation_results.txt", "metric.json"], # metric.json f1:4.818445863775445
    "xquad": ["generation_results_raw.txt", "generation_results.txt", "metric.json"],  # json: acc:0.8453781512605042
    "flores": ["generation_results_.hyp", "spBLEU_summary.csv"], # spBLEU_summary.csv en_zh 0.231234
    "ceval": ["generation_results_raw.txt"] # 最后一行 {"accuracy": 0.18076923076923077}
}

METRIC_RESULT_DICT={
    "xnli": "generation_results.txt", # generation_results.txt 最后一行 {"accuracy": 0.0}
    "mmlu": "generation_results_raw.txt", # 最后一行 {"accuracy": 0.0}
    "mlqa":  "metric.json", # metric.json f1:4.818445863775445， em
    "xquad": "metric.json",  # json: acc:0.8453781512605042
    "flores": "spBLEU_summary.csv", # spBLEU_summary.csv en_zh 0.231234
    "ceval": "generation_results_raw.txt" # 最后一行 {"accuracy": 0.18076923076923077}
}

def json_type_res_txt(path, key=None):
    hf_accuracy = None
    try:
        with open(path, 'r') as f:
            lines = f.readlines()
            i = 0
            for line in reversed(lines):  # 从最后一行开始读取
                try:
                    data = json.loads(line)  # 尝试将行解析为JSON
                    # hf_accuracy = data.get('accuracy', None)  # 尝试从JSON中提取accuracy
                    hf_accuracy=data
                    if hf_accuracy is not None:  # 如果成功提取到accuracy，跳出循环
                        break
                except json.JSONDecodeError:
                    i += 1
                    if i > 4:
                        break
                    continue  # 如果解析JSON失败，忽略这一行并继续下一行
    except Exception as error:
        print(path, "ERROR")
        print(error)
    return hf_accuracy

def json_type_res_json(path, key=None):
    res = None
    try:
        f = json.load(open(path,"r"))
        # res = f[key]
        res = f
    except Exception as error:
        print(path, "ERROR")
        print(error)
    return res

def csv_type_res(path,key=None):
    res = None 
    try:
        f = pd.read_csv(path)
        # res = float(f.values[0][0].split(' ')[1])
        res = f
    except Exception as error:
        print(path, "ERROR")
        print(error)
    
    return res

METRIC_RES_GATHER_FUNC={
    "xnli": [json_type_res_txt, None], # generation_results.txt 最后一行 {"accuracy": 0.0}
    "mmlu": [json_type_res_txt, None], # 最后一行 {"accuracy": 0.0}
    "mlqa":  [json_type_res_json, "f1"], # metric.json f1:4.818445863775445
    "xquad": [json_type_res_json, "acc"],  # json: acc:0.8453781512605042
    "flores": [csv_type_res, None], # spBLEU_summary.csv en_zh 0.231234
    "ceval": [json_type_res_txt, None] # 最后一行 {"accuracy": 0.18076923076923077}
}

# def dlc_run():
#     pass


# def dlc_single_shell_create(save_dir):

#     pass


def dlc_flores_shell_create(config, save_path, lang_pair):
    print(f"flores需要支持：{SUPPORT_MMLU_FLORES}等语言，共计：{2 * len(SUPPORT_MMLU_FLORES) - 1}个推理job.")
    task_name="flores"
    os.makedirs(os.path.join(save_path, task_name), exist_ok=True)
    src_lang, tgt_lang = lang_pair.split('-')
    os.makedirs(os.path.join(save_path, task_name, src_lang+"-"+"x"), exist_ok=True)
    if config.force.lang_pairs is not None:
        for lang_pair in config.force.lang_pairs:
            src_lang, tgt_lang = lang_pair.split('-')
            cur_lang_pair = lang_pair
            os.makedirs(os.path.join(save_path, task_name, src_lang+"-"+"x"), exist_ok=True)
            COMMAND=f'''{PYTHON_PATH} {EVAL_SCRIPT_PATH} --cfg {os.path.join(BASE_CONFIGS_PATH, "flores.yaml")} --lang_pair {cur_lang_pair} --base_model {config.model_path.base_model} --project_path {PROJECT_PATH} --input_file {REVERSE_LGS_DICT[src_lang]}.devtest --subpath {src_lang+"-"+"x"}'''
            NAME=f"auto_inferece_{task_name}_{cur_lang_pair}"
            shell = BASE_SHELL.format_map({
                                    "DLC_CONFIG_PATH": DLC_CONFIG_PATH,
                                    "COMMAND": COMMAND,
                                    "CPFS_PATH": CPFS_PATH,
                                    "NAME": NAME
                                })

            with open(os.path.join(save_path, task_name, src_lang+"-"+"x", cur_lang_pair+".dlc.sh"), "w") as f:
                f.write(shell)
    else:
        for language in SUPPORT_MMLU_FLORES:

            cur_lang_pair = src_lang+ "-" + language
            COMMAND=f'''{PYTHON_PATH} {EVAL_SCRIPT_PATH} --cfg {os.path.join(BASE_CONFIGS_PATH, "flores.yaml")} --lang_pair {cur_lang_pair} --base_model {config.model_path.base_model} --project_path {PROJECT_PATH} --input_file {REVERSE_LGS_DICT[src_lang]}.devtest --subpath {src_lang+"-"+"x"}'''
            NAME=f"auto_inferece_{task_name}_{cur_lang_pair}"
            shell = BASE_SHELL.format_map({
                                    "DLC_CONFIG_PATH": DLC_CONFIG_PATH,
                                    "COMMAND": COMMAND,
                                    "CPFS_PATH": CPFS_PATH,
                                    "NAME": NAME
                                })

            with open(os.path.join(save_path, task_name, src_lang+"-"+"x", cur_lang_pair+".dlc.sh"), "w") as f:
                f.write(shell)
        os.makedirs(os.path.join(save_path, task_name, "x"+"-"+tgt_lang), exist_ok=True)
        for language in SUPPORT_MMLU_FLORES:
            cur_lang_pair = language+ "-" + tgt_lang
            COMMAND=f'''{PYTHON_PATH} {EVAL_SCRIPT_PATH} --cfg {os.path.join(BASE_CONFIGS_PATH, "flores.yaml")} --lang_pair {cur_lang_pair} --base_model {config.model_path.base_model} --project_path {PROJECT_PATH} --input_file {REVERSE_LGS_DICT[language]}.devtest --subpath {"x"+"-"+tgt_lang}'''
            NAME=f"auto_inferece_{task_name}_{cur_lang_pair}"
            shell = BASE_SHELL.format_map({
                                    "DLC_CONFIG_PATH": DLC_CONFIG_PATH,
                                    "COMMAND": COMMAND,
                                    "CPFS_PATH": CPFS_PATH,
                                    "NAME": NAME
                                })

            with open(os.path.join(save_path, task_name, "x"+"-"+tgt_lang, cur_lang_pair+".dlc.sh"), "w") as f:
                f.write(shell)

        if config.task_setting.task_scale == 4:
            print(f"【增加】flores需要支持：{SUPPORT_MMLU_FLORES}等语言，共计：{2 * len(SUPPORT_MMLU_FLORES) - 1}个推理job.")
            os.makedirs(os.path.join(save_path, task_name, "x"+"-"+src_lang), exist_ok=True)
            for language in SUPPORT_MMLU_FLORES:

                cur_lang_pair = language + "-" + src_lang

                COMMAND=f'''{PYTHON_PATH} {EVAL_SCRIPT_PATH} --cfg {os.path.join(BASE_CONFIGS_PATH, "flores.yaml")} --lang_pair {cur_lang_pair} --base_model {config.model_path.base_model} --project_path {PROJECT_PATH} --input_file {REVERSE_LGS_DICT[language]}.devtest --subpath {"x"+"-"+src_lang}'''
                NAME=f"auto_inferece_{task_name}_{cur_lang_pair}"
                shell = BASE_SHELL.format_map({
                                        "DLC_CONFIG_PATH": DLC_CONFIG_PATH,
                                        "COMMAND": COMMAND,
                                        "CPFS_PATH": CPFS_PATH,
                                        "NAME": NAME
                                    })

                with open(os.path.join(save_path, task_name, "x"+"-"+src_lang, cur_lang_pair+".dlc.sh"), "w") as f:
                    f.write(shell)
            os.makedirs(os.path.join(save_path, task_name, tgt_lang+"-"+"x"), exist_ok=True)
            for language in SUPPORT_MMLU_FLORES:
                cur_lang_pair = tgt_lang + "-" +  language
                COMMAND=f'''{PYTHON_PATH} {EVAL_SCRIPT_PATH} --cfg {os.path.join(BASE_CONFIGS_PATH, "flores.yaml")} --lang_pair {cur_lang_pair} --base_model {config.model_path.base_model} --project_path {PROJECT_PATH} --input_file {REVERSE_LGS_DICT[tgt_lang]}.devtest --subpath {tgt_lang+"-"+"x"}'''
                NAME=f"auto_inferece_{task_name}_{cur_lang_pair}"
                shell = BASE_SHELL.format_map({
                                        "DLC_CONFIG_PATH": DLC_CONFIG_PATH,
                                        "COMMAND": COMMAND,
                                        "CPFS_PATH": CPFS_PATH,
                                        "NAME": NAME
                                    })

                with open(os.path.join(save_path, task_name, tgt_lang+"-"+"x", cur_lang_pair+".dlc.sh"), "w") as f:
                    f.write(shell)

    print(f"flores任务的shell脚本准备完毕.")

def dlc_mmlu_shell_create(config, save_path, lang_pair):
    print(f"mmlu需要支持：{SUPPORT_MMLU_FLORES}等语言，共计：{len(SUPPORT_MMLU_FLORES)}个推理job.")
    task_name="mmlu"
    os.makedirs(os.path.join(save_path, task_name), exist_ok=True)


    for language in SUPPORT_MMLU_FLORES:
        COMMAND=f'''{PYTHON_PATH} {EVAL_SCRIPT_PATH} --cfg {os.path.join(BASE_CONFIGS_PATH, "mmlu.yaml")} --lang_pair {language} --base_model {config.model_path.base_model} --project_path {PROJECT_PATH} --data_path /cpfs01/shared/NLP-A100/NLP-A100_hdd/jie/benchmark/mmlu-101/mmlu_all_{language}_zeroshot'''
        NAME=f"auto_inferece_{task_name}_{language}"
        shell = BASE_SHELL.format_map({
                                "DLC_CONFIG_PATH": DLC_CONFIG_PATH,
                                "COMMAND": COMMAND,
                                "CPFS_PATH": CPFS_PATH,
                                "NAME": NAME
                            })

        with open(os.path.join(save_path, task_name, language+".dlc.sh"), "w") as f:
            f.write(shell)
    print(f"mmlu任务的shell脚本准备完毕.")
    
def dlc_mlqa_shell_create(config, save_path, lang_pair):
    if config.task_setting.mlqa_all:
        print(f"mlqa需要支持：{SUPPORT_MLQA}等语言，共计：{len(SUPPORT_MLQA) * len(SUPPORT_MLQA)}个推理job.")
        task_name="mlqa"
        os.makedirs(os.path.join(save_path, task_name), exist_ok=True)
        os.makedirs(os.path.join(save_path, task_name, "mlqa_all"), exist_ok=True)
        for lang1 in SUPPORT_MLQA:
            for lang2 in SUPPORT_MLQA:
                cur_lang_pair = lang1+ "-" + lang2
                COMMAND=f'''{PYTHON_PATH} {EVAL_SCRIPT_PATH} --cfg {os.path.join(BASE_CONFIGS_PATH, "mlqa.yaml")} --lang_pair {cur_lang_pair} --base_model {config.model_path.base_model} --project_path {PROJECT_PATH} --subpath {"mlqa_all"}'''
                NAME=f"auto_inferece_{task_name}_{cur_lang_pair}"
                shell = BASE_MLQA_SHELL.format_map({
                                        "DLC_CONFIG_PATH": DLC_CONFIG_PATH,
                                        "COMMAND": COMMAND,
                                        "CPFS_PATH": CPFS_PATH,
                                        "NAME": NAME
                                    })

                with open(os.path.join(save_path, task_name, "mlqa_all", cur_lang_pair+".dlc.sh"), "w") as f:
                    f.write(shell)
            
    else:
        print(f"mlqa需要支持：{SUPPORT_MLQA}等语言，共计：{2*len(SUPPORT_MLQA)-1}个推理job.")
        task_name="mlqa"
        os.makedirs(os.path.join(save_path, task_name), exist_ok=True)
        src_lang, tgt_lang = lang_pair.split('-')
        os.makedirs(os.path.join(save_path, task_name, src_lang+"-"+"x"), exist_ok=True)
        for language in SUPPORT_MLQA:
            cur_lang_pair = src_lang+ "-" + language

            COMMAND=f'''{PYTHON_PATH} {EVAL_SCRIPT_PATH} --cfg {os.path.join(BASE_CONFIGS_PATH, "mlqa.yaml")} --lang_pair {cur_lang_pair} --base_model {config.model_path.base_model} --project_path {PROJECT_PATH} --subpath {src_lang+"-"+"x"}'''
            NAME=f"auto_inferece_{task_name}_{cur_lang_pair}"
            shell = BASE_MLQA_SHELL.format_map({
                                    "DLC_CONFIG_PATH": DLC_CONFIG_PATH,
                                    "COMMAND": COMMAND,
                                    "CPFS_PATH": CPFS_PATH,
                                    "NAME": NAME
            
                                })

            with open(os.path.join(save_path, task_name, src_lang+"-"+"x", cur_lang_pair+".dlc.sh"), "w") as f:
                f.write(shell)
        os.makedirs(os.path.join(save_path, task_name, "x"+"-"+tgt_lang), exist_ok=True)
        for language in SUPPORT_MLQA:
            cur_lang_pair = language+ "-" + tgt_lang
            COMMAND=f'''{PYTHON_PATH} {EVAL_SCRIPT_PATH} --cfg {os.path.join(BASE_CONFIGS_PATH, "mlqa.yaml")} --lang_pair {cur_lang_pair} --base_model {config.model_path.base_model} --project_path {PROJECT_PATH} --subpath {"x"+"-"+tgt_lang}'''
            NAME=f"auto_inferece_{task_name}_{cur_lang_pair}"
            shell = BASE_MLQA_SHELL.format_map({
                                    "DLC_CONFIG_PATH": DLC_CONFIG_PATH,
                                    "COMMAND": COMMAND,
                                    "CPFS_PATH": CPFS_PATH,
                                    "NAME": NAME
                                })

            with open(os.path.join(save_path, task_name, "x"+"-"+tgt_lang, cur_lang_pair+".dlc.sh"), "w") as f:
                f.write(shell)

        if config.task_setting.task_scale == 4:
            print(f"【增加】mlqa需要支持：{SUPPORT_MLQA}等语言，共计：{2*len(SUPPORT_MLQA)-1}个推理job.")
            os.makedirs(os.path.join(save_path, task_name, "x" +"-"+ src_lang), exist_ok=True)
            for language in SUPPORT_MLQA:
                cur_lang_pair = language + "-" +  src_lang

                COMMAND=f'''{PYTHON_PATH} {EVAL_SCRIPT_PATH} --cfg {os.path.join(BASE_CONFIGS_PATH, "mlqa.yaml")} --lang_pair {cur_lang_pair} --base_model {config.model_path.base_model} --project_path {PROJECT_PATH} --subpath {"x" +"-"+ src_lang}'''
                NAME=f"auto_inferece_{task_name}_{cur_lang_pair}"
                shell = BASE_MLQA_SHELL.format_map({
                                        "DLC_CONFIG_PATH": DLC_CONFIG_PATH,
                                        "COMMAND": COMMAND,
                                        "CPFS_PATH": CPFS_PATH,
                                        "NAME": NAME
                                    })

                with open(os.path.join(save_path, task_name, "x" +"-"+ src_lang, cur_lang_pair+".dlc.sh"), "w") as f:
                    f.write(shell)
            os.makedirs(os.path.join(save_path, task_name, tgt_lang +"-"+ "x"), exist_ok=True)
            for language in SUPPORT_MLQA:
                cur_lang_pair = tgt_lang + "-" + language
                COMMAND=f'''{PYTHON_PATH} {EVAL_SCRIPT_PATH} --cfg {os.path.join(BASE_CONFIGS_PATH, "mlqa.yaml")} --lang_pair {cur_lang_pair} --base_model {config.model_path.base_model} --project_path {PROJECT_PATH} --subpath {tgt_lang +"-"+ "x"}'''
                NAME=f"auto_inferece_{task_name}_{cur_lang_pair}"
                shell = BASE_MLQA_SHELL.format_map({
                                        "DLC_CONFIG_PATH": DLC_CONFIG_PATH,
                                        "COMMAND": COMMAND,
                                        "CPFS_PATH": CPFS_PATH,
                                        "NAME": NAME
                                    })

                with open(os.path.join(save_path, task_name, tgt_lang +"-"+ "x", cur_lang_pair+".dlc.sh"), "w") as f:
                    f.write(shell)

        print(f"mlqa任务的shell脚本准备完毕.")
    
def dlc_xnli_shell_create(config, save_path, lang_pair):
    print(f"xnli需要支持：{SUPPORT_XNLI}等语言，共计：{len(SUPPORT_XNLI)}个推理job.")
    task_name="xnli"
    os.makedirs(os.path.join(save_path, task_name), exist_ok=True)
    for language in SUPPORT_XNLI:
        COMMAND=f'''{PYTHON_PATH} {EVAL_SCRIPT_PATH} --cfg {os.path.join(BASE_CONFIGS_PATH, "xnli.yaml")} --lang_pair {language} --base_model {config.model_path.base_model} --project_path {PROJECT_PATH}'''
        NAME=f"auto_inferece_{task_name}_{language}"
        shell = BASE_SHELL.format_map({
                                "DLC_CONFIG_PATH": DLC_CONFIG_PATH,
                                "COMMAND": COMMAND,
                                "CPFS_PATH": CPFS_PATH,
                                "NAME": NAME
                            })
        
        with open(os.path.join(save_path, task_name, language+".dlc.sh"), "w") as f:
            f.write(shell)
    print(f"xnli任务的shell脚本准备完毕.")
    
def dlc_xquad_shell_create(config, save_path,lang_pair):
    print(f"xquad需要支持：{SUPPORT_XQUAD}等语言，共计：{len(SUPPORT_XQUAD)}个推理job.")
    task_name="xquad"
    os.makedirs(os.path.join(save_path, task_name), exist_ok=True)
    for language in SUPPORT_XQUAD:
        COMMAND=f'''{PYTHON_PATH} {EVAL_SCRIPT_PATH} --cfg {os.path.join(BASE_CONFIGS_PATH, "xquad.yaml")} --lang_pair {language} --base_model {config.model_path.base_model} --project_path {PROJECT_PATH}'''
        NAME=f"auto_inferece_{task_name}_{language}"
        shell = BASE_SHELL.format_map({
                                "DLC_CONFIG_PATH": DLC_CONFIG_PATH,
                                "COMMAND": COMMAND,
                                "CPFS_PATH": CPFS_PATH,
                                "NAME": NAME
                            })
        
        with open(os.path.join(save_path, task_name, language+".dlc.sh"), "w") as f:
            f.write(shell)
    print(f"xquad任务的shell脚本准备完毕.")
    
def dlc_ceval_shell_create(config, save_path,lang_pair):
    print(f"ceval需要支持：1 语言，共计：1个推理job.")
    task_name="ceval"
    os.makedirs(os.path.join(save_path, task_name), exist_ok=True)
    for language in ['zh']:
        COMMAND=f'''{PYTHON_PATH} {EVAL_SCRIPT_PATH} --cfg {os.path.join(BASE_CONFIGS_PATH, "ceval.yaml")} --lang_pair {language} --base_model {config.model_path.base_model} --project_path {PROJECT_PATH} --data_path /cpfs01/shared/NLP-A100/NLP-A100_hdd/jie/benchmark/c-Eval/ceval_all_{language}_zeroshot'''
        NAME=f"auto_inferece_{task_name}_{language}"
        shell = BASE_SHELL.format_map({
                                "DLC_CONFIG_PATH": DLC_CONFIG_PATH,
                                "COMMAND": COMMAND,
                                "CPFS_PATH": CPFS_PATH,
                                "NAME": NAME
                            })
        
        with open(os.path.join(save_path, task_name, language+".dlc.sh"), "w") as f:
            f.write(shell)
    print(f"ceval任务的shell脚本准备完毕.")
    

def count_sh_files(directory):
    count = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if fnmatch.fnmatch(file, '*.sh'):
                count += 1
    return count

def get_sh_file_paths(directory):
    sh_file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if fnmatch.fnmatch(file, '*.sh'):
                file_path = os.path.join(root, file)
                sh_file_paths.append(file_path)
    return sh_file_paths

def get_sh_file_paths2(directory, task_names):
    sh_file_paths = {}
    for task in task_names:
        sh_file_paths[task] = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if fnmatch.fnmatch(file, '*.sh'):
                file_path = os.path.join(root, file)
                for search_task in task_names:
                    if search_task in root:
                        task = search_task
                        break
                sh_file_paths[task].append(file_path)
    return sh_file_paths

def dlc_batch_shell_create(config):
    
    print(f"开始创建shell脚本存储路径")
    model_path = config.model_path.base_model if config.model_path.lora is None else config.model_path.lora
    # 获取当前日期
    current_datetime = datetime.now() #dt.date.today()
    # 根据日期创建文件夹名称
    # folder_name = current_date.strftime("%Y-%m-%d")
    folder_name = current_datetime.strftime("%Y-%m-%d-%H-%M-%S")
    save_path = os.path.join(model_path, config.output.summary_dir, folder_name+"."+config.task_setting.lang_pair, "inference.shell_cache")
    os.makedirs(save_path, exist_ok=True)
    print(f"shell脚本存储路径：{save_path}")
    task_func = {
        "xnli": dlc_xnli_shell_create,
        "ceval": dlc_ceval_shell_create,
        "xquad": dlc_xquad_shell_create,
        "mlqa": dlc_mlqa_shell_create,
        "mmlu": dlc_mmlu_shell_create,
        "flores": dlc_flores_shell_create
    }
    
    
    
    task_names = config.task_setting.tasks
    print(f"当前需要推理的任务有：{task_names}")
    for task in task_names:
        task_func[task](config, save_path, config.task_setting.lang_pair)
    print(f"建立子任务推理脚本！")
    
    # 统计以 .sh 结尾的文件个数
    sh_file_count = count_sh_files(save_path)
    print(f"Total .sh files: {sh_file_count}")
    
    config.save_path = save_path
    config.save_base_path = os.path.join(model_path, config.output.summary_dir, folder_name+"."+config.task_setting.lang_pair)
    

def check_finish(config):
    summary_path = os.path.join(config.save_base_path, "task_job_states.csv")
    df = pd.read_csv(summary_path)
    df['startTime'] = pd.to_datetime(df['startTime'])
    df[["metric"]] = df[["metric"]].astype(str)
    
    for index, row in df.iterrows():
        if row['state'] == 'running' or row['state'] == 'waiting':
            result_file_path = os.path.join(row['result_path'], row['metric_file'])
            if os.path.exists(result_file_path):
                task_name = row['task']
                parser_func, key = METRIC_RES_GATHER_FUNC[task_name]
                # if task_name == "xquad":
                #     score = parser_func(result_file_path, "acc")
                # elif task_name == "mlqa":
                #     score = parser_func(result_file_path, "f1")
                # else:
                score = parser_func(result_file_path, None)
                # print(f"获取到的结果为：{score}")
                # print(f"result_file_path:{result_file_path}")
                if score is not None:
                    df.at[index, 'state'] = 'finish'
                    df.at[index, 'metric'] = str(score)
                if datetime.now() - row['startTime'] > timedelta(hours=24) and df.at[index, 'state'] == 'running':
                    df.at[index, 'state'] = 'failRun'
    df.to_csv(summary_path, index=False)

def check_running(config, max_running_jobs):
    summary_path = os.path.join(config.save_base_path, "task_job_states.csv")
    df = pd.read_csv(summary_path)
    
    running_jobs = df[df['state'] == 'running'].shape[0]
    if running_jobs < max_running_jobs:
        waiting_jobs = df[df['state'] == 'waiting']
        for _, job in waiting_jobs.iterrows():
            shell_script = job['shell']
            return_code = call("sudo sh " + shell_script, shell=True)
            if return_code == 0:
                df.loc[df['shell'] == shell_script, 'state'] = 'running'
                df.loc[df['shell'] == shell_script, 'startTime'] = datetime.now().strftime('%Y-%m-%d %H:%M')
            else:
                df.loc[df['shell'] == shell_script, 'state'] = 'failStart'
            if df[df['state'] == 'running'].shape[0] >= max_running_jobs:
                break
    df.to_csv(summary_path, index=False)

def build_run_schedule(config):
    print("--------------------------------------- 开始 ----------------------------------------------")
    
    # 获取所有以 .sh 结尾的文件路径
    sh_file_paths = get_sh_file_paths2(config.save_path, config.task_setting.tasks)
    for k, v in sh_file_paths.items():
        print(f"任务{k}, 有{len(v)}个")
    # print(sh_file_paths)
    print(f"开始构建全部任务的进度表")
    # 初始化空的表格
    df = pd.DataFrame(columns=["task", "lang_pair", "state", "prefix_path", "result_path", "copy_files", "metric_file", "shell", "startTime", "endTime", "metric"])
    
    
    task_jobs = []
    model_path = config.model_path.base_model if config.model_path.lora is None else config.model_path.lora
    for task, jobs in sh_file_paths.items():
        for job in jobs:
            lang_pair = job.split('/')[-1].split('.')[0]
            prefix_path = job.split('/')[-2]
            if prefix_path == task:
                prefix_path = None
            
            result_path = os.path.join(model_path,task,prefix_path,lang_pair) if prefix_path is not None else os.path.join(model_path,task,lang_pair)
            copy_files = COPY_CONFIG_DICT[task]
            task_jobs.append([task, lang_pair, "waiting", prefix_path, result_path, copy_files, METRIC_RESULT_DICT[task], job, None, None, None]) 
    summary_path = os.path.join(config.save_base_path,"task_job_states.csv")
    os.makedirs(config.save_base_path, exist_ok=True)
    # 构建 DataFrame
    df = pd.DataFrame(task_jobs, columns=["task", "lang_pair", "state", "prefix_path", "result_path", "copy_files", "metric_file","shell", "startTime", "endTime", "metric"])
    
    # 保存 DataFrame 到文件
    df.to_csv(summary_path, index=False)
    
def get_args(parser):
    parser.add_argument('--cfg', type=str, default="/cpfs01/shared/NLP-A100/NLP-A100_hdd/jie/multilingual_LLM/configs/A100/total_inference.yaml", help='model name in the hub or local path')
    parser.add_argument('--model_path', type=str, default=None)
    # parser.add_argument('--refresh', action="store_true")
    return parser.parse_args()

    
def main():
    global EVAL_SCRIPT_PATH
    parser = argparse.ArgumentParser()
    args = get_args(parser)
     # 加载测试参数
    config = read_yaml_file(args.cfg)
    config = AttributeDict.map_nested_dicts(config)
    # 修改 torch_dtype
    EVAL_SCRIPT_PATH = EVAL_SCRIPT_PATH + f" --torch_dtype {config.model_path.torch_dtype}"
    # 修改lora参数
    if config.model_path.lora is not None:
        EVAL_SCRIPT_PATH = EVAL_SCRIPT_PATH + f" --lora {config.model_path.lora}"
        
    if args.model_path is not None:
        config.model_path.base_model = args.model_path

    # dlc_run()
    # dlc_single_shell_create(save_dir)
    dlc_batch_shell_create(config)
    build_run_schedule(config)
    
    max_running_jobs = config.task_setting.max_running_jobs
    exe_count = 0.
    # if not args.refresh:
    print("启动前先检查一次已经存在的一些结果")
    check_finish(config)
    # else:
    #     print(f"忽略存在的结果，直接执行！")
    while True:
        print(f"--------------------------- 程序正在遍历中:{exe_count} --------------------------------- ")
        # 每隔1分钟检查一次任务的状态
        # if exe_count!=0:
        check_finish(config)
        check_running(config, max_running_jobs)

        # 读取进度表
        summary_path = os.path.join(config.save_base_path, "task_job_states.csv")
        df = pd.read_csv(summary_path)
        # 统计任务状态
        waiting_jobs = df[df['state'] == 'waiting'].shape[0]
        running_jobs = df[df['state'] == 'running'].shape[0]
        finish_jobs = df[df['state'] == 'finish'].shape[0]
        failRun_jobs = df[df['state'] == 'failRun'].shape[0]
        failStart_jobs = df[df['state'] == 'failStart'].shape[0]

        # 输出任务状态
        print(f"等待任务数: {waiting_jobs}")
        print(f"运行任务数: {running_jobs}")
        print(f"成功任务数: {finish_jobs}")
        print(f"运行失败任务数: {failRun_jobs}")
        print(f"启动失败任务数: {failStart_jobs}")

        # 如果所有任务的状态都不是 'waiting'，则退出循环
        if 'waiting' not in df['state'].values and 'running' not in df['state'].values:
            break
        
        # 等待十二分钟
        time.sleep(720)
        exe_count+=1

if __name__ == "__main__":
    main()