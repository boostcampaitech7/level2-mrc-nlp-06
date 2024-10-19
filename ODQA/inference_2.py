"""
Open-Domain Question Answering 을 수행하는 inference 코드 입니다.

대부분의 로직은 train.py 와 비슷하나 retrieval, predict 부분이 추가되어 있습니다.
"""

import logging
import sys
import os
import json
sys.path.append('/data/ephemeral/home/jh/level2-mrc-nlp-06/ODQA')
from typing import Callable, Dict, List, NoReturn, Tuple
import argparse
import numpy as np
import pandas as pd

# ========= Hugging Face ========= #
from datasets import (
    Dataset,
    DatasetDict,
    Features,
    Sequence,
    Value,
    load_from_disk,
    load_metric,
)
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)

# ========= Modules ========= #
from modules.extractiveQA import ExtractiveQA
from modules.spraseRetrieval import Sparse_Model
from utils_common import json_to_config


logger = logging.getLogger(__name__)


def main(args, model_args, training_args, data_args, datasets):
    # AutoConfig를 이용하여 pretrained model 과 tokenizer를 불러옵니다.
    # argument로 원하는 모델 이름을 설정하면 옵션을 바꿀 수 있습니다.
    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name
        else model_args.model_name_or_path,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        use_fast=True,
    )
    if model_args.generation:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config
        )
    else:
        model = AutoModelForQuestionAnswering.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
        )

def run_odqa(args, inference_config, test_datasets, wiki_path, valid_datasets=None):
    
    # Load QA Model
    match args.qa:
        case "ext":
            QA_model = ExtractiveQA(args, inference_config.qa_config, test_datasets, valid_datasets)
        case "abs":
            pass

    # Load Retrieval Model
    match args.retrieval:
        case "sparse":
            Ret_model = Sparse_Model(args, inference_config, test_datasets, wiki_path, valid_datasets)
        case "dense":
            pass
        case "hybrid":
            pass

    # Inference
    output_dir = os.path.join(inference_config.output_dir, "test") # predictions.json 출력폴더

    # 1. Retrieval Model에서 Question에 맞는 Context를 가져옴
    test_retrieve_datasets = Ret_model.get_contexts(test_datasets)

    # 2. Reader에 Context를 전달하여 Inference를 수행
    # Extraction Reader의 경우 postprocess_qa_predictions함수를 수행하면서 자동적으로 Json 저장
    test_predictions = QA_model.predict(test_retrieve_datasets, output_dir)

    # 3. 만약에 Validation도 추가로 Predictions을 뽑고 싶을 때
    if valid_datasets is not None:
        output_dir = os.path.join(inference_config.output_dir, "validation")
        valid_retrieve_datasets = Ret_model.get_contexts(valid_datasets)
        valid_predictions = QA_model.predict(valid_retrieve_datasets, output_dir)

    




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Inference Options 설정", default="./config/base_config.json")
    parser.add_argument("--pred_dir", help="Prediction 폴더 설정", default="./predictions")
    parser.add_argument("--qa", required=True, help="qa 타입을 설정해주세요. ['abs','ext']")
    parser.add_argument("--retrieval", required=True, help="retrieval 타입을 설정해주세요. ['sparse','hybrid','dense']")
    parser.add_argument("--do_valid", action="store_true", help="Validation 데이터셋에 대해 Prediction을 저장하기 위한 옵션")
    args = parser.parse_args()
    
    print("=" * 8)
    print("QA Model type :",args.qa)
    print("Retrieval Model type :",args.retrieval)
    print("Validation Predictions Options", args.do_valid)
    print("=" * 8)

    # logging 설정
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    inference_config = json_to_config(args.config)
    test_path = os.path.join(inference_config.datasets,"test")
    wiki_path = os.path.join(inference_config.datasets,"wiki.json")

    if args.do_valid:
        valid_path = os.path(inference_config.datasets,"validation")
        test_datasets = load_from_disk(test_path)
        valid_datasets = load_from_disk(valid_path)
        answers = {data["id"]:data["answers"]["text"][0] for data in valid_datasets["validation"]}
        valid_datasets = valid_datasets.remove_columns(["title","context","answers","document_id","__index_level_0__"])
        run_odqa(args, inference_config, test_datasets, wiki_path, valid_datasets=valid_datasets)
    else:
        test_datasets = load_from_disk(test_path)
        run_odqa(args, inference_config, test_datasets, wiki_path)
    
    print("******* predict dataset Predictions *******")

    
    
    # if args.do_valid:

    #     # Validation Dataset 대해서도 Predictions 수행
    #     output_dir = args.pred_dir

    #     # logging 설정
    #     logging.basicConfig(
    #         format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    #         datefmt="%m/%d/%Y %H:%M:%S",
    #         handlers=[logging.StreamHandler(sys.stdout)],
    #     )
    #     # verbosity 설정 : Transformers logger의 정보로 사용합니다 (on main process only)
    #     logger.info("Training/evaluation parameters %s", training_args)

    #     # 모델을 초기화하기 전에 난수를 고정합니다.
    #     set_seed(training_args.seed)

    #     datasets = load_from_disk(data_args.train_dataset_name)
    #     answers = {data["id"]:data["answers"]["text"][0] for data in datasets["validation"]}
    #     datasets = datasets.remove_columns(["title","context","answers","document_id","__index_level_0__"])
    #     print("******* validation dataset Predictions *******")
    #     print(datasets)
    #     main(args, model_args, training_args, data_args, datasets)

    #     # 기존 predictions 불러와서 정답 추가
    #     dict_to_json(os.path.join(training_args.output_dir,"predictions.json"),
    #                  os.path.join(training_args.output_dir,"prediction_with_ans.json"),
    #                  answers)