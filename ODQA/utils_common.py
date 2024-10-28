import yaml
from box import Box
import json
import os

from datasets import (
    Dataset,
    DatasetDict,
    Features,
    Sequence,
    Value
)

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

def df_to_dataset(df, do_predict, do_eval):
    if do_predict:
        f = Features(
            {
                "context": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
            }
        )
    elif do_eval:
        f = Features(
            {
                "context": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "answers": Sequence(
                    feature={
                        "text": Value(dtype="string", id=None),
                        "answer_start": Value(dtype="int64", id=None),
                    },
                    length=-1,
                    id=None,
                ),
            }
        )
    datasets = DatasetDict({"validation": Dataset.from_pandas(df, features=f)})
    return datasets