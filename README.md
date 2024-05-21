## How-Multilingual-is-LLaMA

Codes and data for: **How Multilingual is LLaMA?**

⚠️ *Under double-blind peer review, full implementations and data will be released later.*🚧

[[Paper](https://arxiv.org/abs/2311.09071)]

# Introduction

```
pip install -r requirements.txt
```
## Usage

We develope our training pipeline based on the [Parrot](https://github.com/wxjiao/ParroT) repository. 


* Run Full Tuning
```bash
bash ./training_scripts/train_full_ft.sh
```

* Run Embed Tuning
```bash
bash ./training_scripts/train_embed_ft.sh
```

* Run Lora Tuning
```bash
bash ./training_scripts/train_lora_ft.sh
```

* Run Full Tuning with extending vocabulary
```bash
bash ./training_scripts/train_extend_vocab.sh
```

* Run training with shortening subword sequences
```bash
bash ./Shorten/train_with_shorten.sh
```

* Run inference on Flores
```bash
bash ./evaluation/test_flores.sh
```





## Reference

If you are interested in our work, please use the following citation format when referencing our paper:

```bibtex
@misc{yuan2023multilingual,
      title={How Multilingual is Multilingual LLM?}, 
      author={Fei Yuan and Shuai Yuan and Zhiyong Wu and Lei Li},
      year={2023},
      eprint={2311.09071},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

---
