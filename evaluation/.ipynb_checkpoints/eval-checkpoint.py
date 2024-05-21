import yaml
import os
import json
import argparse
import importlib.util
import getpass

def read_yaml_file(file_path):
    with open(file_path, 'r') as file:
        try:
            return yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
            
class YamlReader():

    def __init__(self, yamlf):

        if os.path.exists(yamlf):
            self.yamlf = yamlf
        else:
            raise FileExistsError("The file does not exist!")
        self._data = None  

    @property
    def data(self):
        
        if not self._data:  
            with open(self.yamlf, 'rb') as f:
                self._data = list(yaml.safe_load_all(f))  
        return self._data[0]

class AttributeDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttributeDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    @staticmethod
    def map_nested_dicts(ob):
        if isinstance(ob, dict):
            return AttributeDict({k: AttributeDict.map_nested_dicts(v) for k, v in ob.items()})
        elif isinstance(ob, list):
            return [AttributeDict.map_nested_dicts(i) for i in ob]
        else:
            return ob

def load_module_from_path(path):
    spec = importlib.util.spec_from_file_location("module.name", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def get_args(parser):
    parser.add_argument('--cfg', type=str, default="./configs/xnli.yaml", help='model name in the hub or local path')
    parser.add_argument('--base_model', type=str, default=None, help='model name in the hub or local path')
    parser.add_argument('--lora', type=str, default=None, help='model name in the hub or local path')
    parser.add_argument('--data_path', type=str, default=None, help='model name in the hub or local path')
    parser.add_argument('--lang_pair', type=str, default=None, help='model name in the hub or local path')
    parser.add_argument('--batch_size', type=int, default=None, help='model name in the hub or local path')
    parser.add_argument('--input_file', type=str, default=None, help='model name in the hub or local path')
    parser.add_argument('--target_column', type=str, default=None, help='model name in the hub or local path')
    parser.add_argument('--prefix_column', type=str, default=None, help='model name in the hub or local path')
    parser.add_argument('--output_file_prefix', type=str, default=None, help='model name in the hub or local path')
    parser.add_argument('--subpath', type=str, default=None, help='model name in the hub or local path')
    parser.add_argument('--postprocess', type=str, default=None, help='model name in the hub or local path')
    parser.add_argument('--torch_dtype', type=str, default=None, help='model name in the hub or local path')
    parser.add_argument('--project_path', type=str, default="", help='model name in the hub or local path')
    parser.add_argument('--beam_size', type=int, default=1, help='model name in the hub or local path')
    parser.add_argument('--rpd', type=str, default=1, help='model name in the hub or local path')
    
    return parser.parse_args()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = get_args(parser)
    
    config = read_yaml_file(args.cfg)
    config = AttributeDict.map_nested_dicts(config)
    
    if args.base_model is not None:
        config.model_path.base_model = args.base_model
    if args.lora is not None:
        config.model_path.lora = args.lora
    if args.data_path is not None:
        config.dataset.path = args.data_path
    if args.lang_pair is not None:
        config.dataset.lang_pair = args.lang_pair
    if args.batch_size is not None:
        config.generat.batch_size = args.batch_size
    if args.target_column is not None:
        config.generat.target_column = args.target_column
    if args.prefix_column is not None:
        config.generat.prefix_column = args.prefix_column
    if args.input_file is not None:
        config.dataset.input_file = args.input_file
    if args.output_file_prefix is not None:
        config.output.output_file_prefix = args.output_file_prefix
    if args.subpath is not None:
        config.output.subpath = args.subpath
    if args.postprocess is not None:
        config.generat.postprocess = args.postprocess
    if args.torch_dtype is not None:
        config.model_path.torch_dtype = args.torch_dtype
    if args.beam_size is not None:
        config.generat.beam_size=args.beam_size
    if args.rpd is not None:
        config.rpd=args.rpd
        


    
    print(f"Language Pair：{args.lang_pair}")
    data_process_func =  load_module_from_path(f"{args.project_path}src/datasets/{config.dataset.loader}.py").dataloader
    test_process_func = load_module_from_path(f"{args.project_path}src/tasks/{config.dataset.loader}.py").test_process
    
    data_dict = data_process_func(config)
    test_process_func(config, data_dict)
    # metric 