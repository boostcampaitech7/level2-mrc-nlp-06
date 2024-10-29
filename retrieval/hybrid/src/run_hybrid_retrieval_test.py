import json
import os, sys
import pickle
import time
import random
from tqdm.auto import tqdm
from contextlib import contextmanager
from typing import List, NoReturn, Optional, Tuple, Union

import argparse
from jsonargparse import ArgumentParser, ActionConfigFile

import numpy as np
import pandas as pd

import torch
from datasets import Dataset, concatenate_datasets, load_from_disk
from transformers import AutoTokenizer, AutoModel


sys.path.append(os.path.abspath("../utils/"))
from utils import *

from hybrid_retrieval import HybridRetrieval

seed = 2024
random.seed(seed)  # python random seed 고정
np.random.seed(seed)  # numpy random seed 고정

if __name__ == "__main__":

    parser = ArgumentParser(description="")

    parser.add_argument(
        "-c",
        "--config",
        action=ActionConfigFile,
        help="this config file is hybrid_retrieval.json in ../config/ foler.",
    )  # action="store"
    parser.default_config_files = ["../config/hybrid_retrieval.json"]

    parser.add_argument(
        "--wiki_path", metavar="wikipedia_documents.json", type=str, help=""
    )
    parser.add_argument("--data_path", metavar="./data/", type=str, help="")
    parser.add_argument("--train_folder", metavar="train_dataset/", type=str, help="")
    parser.add_argument("--eval_folder", metavar="eval_dataset/", type=str, help="")
    parser.add_argument(
        "--model_name_or_path",
        metavar="bert-base-multilingual-cased",
        type=str,
        help="",
    )
    parser.add_argument("--retriever_type", metavar="two_stage", type=str, help="")
    parser.add_argument("--alpha", metavar=0.5, type=float, help="")
    parser.add_argument("--use_faiss", metavar=False, type=bool, help="")
    parser.add_argument("--topk", metavar=1, type=int, help="")
    parser.add_argument("--test", metavar=False, type=bool, help="")
    parser.add_argument("--test_ks", metavar=[1, 2], type=list, help="")

    args = parser.parse_args()
    model_args = args

    config = args
    wiki_path = config.wiki_path
    data_path = config.data_path
    train_folder = config.train_folder
    eval_folder = config.eval_folder
    model_name_or_path = config.model_name_or_path
    retriever_type = config.retriever_type
    alpha = config.alpha
    device = "cuda" if torch.cuda.is_available() else "cpu"
    topk = config.topk
    test = config.test
    test_ks = config.test_ks

    print(config)

    # Test sparse
    print("load dataset")
    org_dataset = load_from_disk(os.path.join(data_path, train_folder))
    full_ds = concatenate_datasets(
        [
            org_dataset["train"].flatten_indices(),
            org_dataset["validation"].flatten_indices(),
        ]
    )  # train dev 를 합친 4192 개 질문에 대해 모두 테스트
    print("*" * 40, "query dataset", "*" * 40)
    print(full_ds)

    retriever = HybridRetrieval(
        model_args=config,
        data_path=data_path,
        context_path=wiki_path,
    )

    pd.set_option("display.max_columns", None)

    query = "대통령을 포함한 미국의 행정부 견제권을 갖는 국가 기관은?"  # "샤이닝 폼을 무엇이라고 칭하기도 하나요?"#"대통령을 포함한 미국의 행정부 견제권을 갖는 국가 기관은?"

    print("not faiss")
    # for topk in test_ks:
    #    retriever.test(retriever_type="sparse", query_or_dataset=full_ds, topk=topk)
    #    retriever.retrieve(retriever_type="sparse", query_or_dataset=query, topk=topk)

    if test:
        for alpha in range(1, 10, 2):
            for topk in test_ks:
                retriever.alpha = alpha / 10

                # retriever.test(retriever_type="bm25", query_or_dataset=full_ds, topk=topk)   # full_ds
                # retriever.retrieve(retriever_type="bm25", query_or_dataset=query, topk=topk) # single query

                # retriever.test(retriever_type="bm25s_two_stage", query_or_dataset=full_ds, topk=topk)   # full_ds
                # retriever.retrieve(retriever_type="bm25s_two_stage", query_or_dataset=query, topk=topk) # single query

                # retriever.test(retriever_type="sparse", query_or_dataset=full_ds, topk=topk)   # full_ds
                # retriever.retrieve(retriever_type="sparse", query_or_dataset=query, topk=topk) # single query

                # retriever.test(retriever_type="dense", query_or_dataset=full_ds, topk=topk)
                # retriever.retrieve(retriever_type="dense", query_or_dataset=query, topk=topk)

                retriever.test(
                    retriever_type="hybrid", query_or_dataset=full_ds, topk=topk
                )
                retriever.retrieve(
                    retriever_type="hybrid", query_or_dataset=query, topk=topk
                )

                # retriever.test(retriever_type="two_stage", query_or_dataset=full_ds, topk=topk)
                # retriever.retrieve(retriever_type="two_stage", query_or_dataset=query, topk=topk)

                # retriever.test(retriever_type="bm25_two_stage", query_or_dataset=full_ds, topk=topk)
                # retriever.retrieve(retriever_type="bm25_two_stage", query_or_dataset=query, topk=topk)
