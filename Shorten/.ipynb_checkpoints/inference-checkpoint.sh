PROJ_PATH=/cpfs01/user/yuanshuai/code/how_multi
LANG=te

sudo /cpfs01/shared/NLP-A100/NLP-A100_hdd/jie/anaconda3/envs/llm2/bin/python ${PROJ_PATH}/Shorten/inference.py --cfg ${PROJ_PATH}/Shorten/flores.yaml --input_file eng.devtest --lang_pair en-${LANG} --base_model /cpfs01/user/yuanshuai/code/tokenize/finetuned/compress/llama-7b-hf.80000.en_${LANG}_bilingual_alpaca.json.compress.single_227 --decode_map ${PROJ_PATH}/Shorten/decode_map/en-${LANG}_rm1_map.json --beam_size 4