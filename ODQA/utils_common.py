import yaml
from box import Box
import json
import os

def configurer(config_path):
    with open(config_path) as file:
        config = yaml.safe_load(file)
        config = Box(config)

    return config

def dict_to_json(json_path, save_path, answer):
    # json 불러오기

    with open(json_path, "r") as f:
        data = json.load(f)
    
    formated_data = {}
    for i,id in enumerate(data.keys()):
        formated_data[id] = {"predict":data[id],"answers":answer[id]}

    with open(save_path,"w", encoding="utf-8") as f:
        json.dump(formated_data, f, ensure_ascii=False, indent=4)

def json_to_config(json_file):
    
    with open(json_file, "r") as f:
        args_dicts = json.load(f)
    args_box = Box(args_dicts)

    return args_box
