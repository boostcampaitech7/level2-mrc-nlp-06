"""
Open-Domain Question Answering 을 수행하는 inference 코드 입니다.

대부분의 로직은 train.py 와 비슷하나 retrieval, predict 부분이 추가되어 있습니다.
"""

import logging
import sys
import os
import json
sys.path.append('/data/ephemeral/home/jh/level2-mrc-nlp-06/ODQA')
sys.path.append('/data/ephemeral/home/jh/level2-mrc-nlp-06/reader')
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
from modules.denseRetrieval import Dense_Model
from utils_common import json_to_config, dict_to_json


logger = logging.getLogger(__name__)

def run_odqa(args, inference_config, test_datasets, wiki_path, pred_dir, valid_datasets=None):
    
    # Load QA Model
    match args.qa:
        case "ext":
            print("******* Extractive QA Model 선택 *******")
            QA_model = ExtractiveQA(args, inference_config.qa_config, test_datasets, valid_datasets)
        case "abs":
            pass

    # Load Retrieval Model
    match args.retrieval:
        case "sparse":
            print("******* Sparse Retrieval Model 선택 *******")
            Ret_model = Sparse_Model(inference_config, test_datasets, wiki_path, valid_datasets)
        case "dense":
            print(("******* Dense (SBERT) Retrieval Model 선택 *******"))
            Ret_model = Dense_Model(inference_config, test_datasets, wiki_path, valid_datasets)
            pass
        case "hybrid":
            pass

    # Inference
    output_dir = os.path.join(pred_dir, "test") # predictions.json 출력폴더

    # 1. Retrieval Model에서 Question에 맞는 Context를 가져옴
    test_retrieve_datasets = Ret_model.get_contexts(test_datasets)

    # 2. Reader에 Context를 전달하여 Inference를 수행
    # Extraction Reader의 경우 postprocess_qa_predictions함수를 수행하면서 자동적으로 Json 저장
    test_predictions = QA_model.predict(test_retrieve_datasets, output_dir)

    # 3. 만약에 Validation도 추가로 Predictions을 뽑고 싶을 때
    if valid_datasets is not None:
        output_dir = os.path.join(pred_dir, "validation")
        valid_retrieve_datasets = Ret_model.get_contexts(valid_datasets)
        valid_predictions = QA_model.predict(valid_retrieve_datasets, output_dir)
        dict_to_json(os.path.join(output_dir,"predictions.json"),
                     os.path.join(output_dir,"prediction_with_ans.json"),
                     answers)

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
    wiki_path = os.path.join(inference_config.datasets,"wikipedia_documents.json")
    
    # datasets load
    print("Test Dataset Load --------")
    datasets = load_from_disk(inference_config.datasets)
    test_datasets = datasets["test"]
    print(test_datasets)

    if args.do_valid:
        print("validation Dataset Load --------")
        valid_datasets = datasets["validation"]
        answers = {data["id"]:data["answers"]["text"][0] for data in valid_datasets}
        valid_datasets = valid_datasets.remove_columns(["title","context","answers","document_id","__index_level_0__"])
        print(valid_datasets)
        run_odqa(args, inference_config, test_datasets, wiki_path, args.pred_dir, valid_datasets=valid_datasets)
    else:
        test_datasets = datasets["test"]
        run_odqa(args, inference_config, test_datasets, wiki_path, args.pred_dir)
    
    print("******* predict dataset Predictions *******")
