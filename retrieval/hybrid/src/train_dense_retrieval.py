import json
import os, sys
import pickle
import time
import random
from tqdm.auto import tqdm, trange
from contextlib import contextmanager
from typing import List, NoReturn, Optional, Tuple, Union
from pprint import pprint

from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk
from transformers import TrainingArguments
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModel,
    AutoModelForQuestionAnswering,
)
from transformers import BertModel, BertPreTrainedModel, BertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup

import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

sys.path.append(os.path.abspath("../utils/"))
from utils import *

from bi_encoder import BiRetrieval, BertEncoder

from jsonargparse import ArgumentParser, ActionConfigFile

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
    parser.default_config_files = ["../config/dense_retrieval.json"]

    parser.add_argument(
        "--wiki_path", metavar="wikipedia_documents.json", type=str, help=""
    )
    parser.add_argument("--data_path", metavar="./data/", type=str, help="")
    parser.add_argument("--train_folder", metavar="train_dataset/", type=str, help="")
    parser.add_argument("--eval_folder", metavar="eval_dataset/", type=str, help="")
    parser.add_argument(
        "--model_name_or_path", metavar="klue/bert-base", type=str, help=""
    )

    parser.add_argument("--train", metavar=True, type=bool, help="")
    parser.add_argument("--train_samples", metavar=True, type=bool, help="")
    parser.add_argument("--num_samples", metavar=10, type=int, help="")
    parser.add_argument("--batch_size", metavar=8, type=int, help="")
    parser.add_argument("--num_neg", metavar=2, type=int, help="")
    parser.add_argument("--epochs", metavar=1, type=int, help="")
    parser.add_argument("--learning_rate", metavar=2e-5, type=float, help="")
    parser.add_argument("--weight_decay", metavar=1e-2, type=float, help="")

    parser.add_argument("--evaluate", metavar=True, type=bool, help="")

    parser.add_argument("--topk", metavar=1, type=int, help="")
    parser.add_argument("--test_ks", metavar=[1, 2], type=list, help="")

    args = parser.parse_args()
    model_args = args
    config = args

    wiki_path = config.wiki_path
    data_path = config.data_path
    train_folder = config.train_folder
    eval_folder = config.eval_folder
    model_name_or_path = config.model_name_or_path

    train = config.train
    train_samples = config.train_samples
    num_samples = config.num_samples
    batch_size = config.batch_size
    num_neg = config.num_neg
    epochs = config.epochs
    learning_rate = config.learning_rate
    weight_decay = config.weight_decay

    evaluate = config.evaluate

    topk = config.topk
    test_ks = config.test_ks

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(config)

    train_dataset_path = data_path + train_folder

    # dataset = load_dataset("squad_kor_v1")["train"]
    # dataset = json.load(wikipedia_path)["train"]

    # Test sparse
    print("load dataset")
    org_dataset = load_from_disk(train_dataset_path)
    full_ds = concatenate_datasets(
        [
            org_dataset["train"].flatten_indices(),
            # org_dataset["validation"].flatten_indices(),
        ]
    )  # train dev 를 합친 4192 개 질문에 대해 모두 테스트
    print("*" * 40, "query dataset", "*" * 40)
    print(full_ds)
    dataset = full_ds

    sample_idx = np.random.choice(range(len(dataset)), num_samples)
    dataset = dataset[sample_idx] if train_samples else dataset
    print(len(dataset["question"]))

    print("model_path:", model_name_or_path)

    args = TrainingArguments(
        output_dir="dense_retrieval",
        evaluation_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=weight_decay,
    )

    print("bi encoder start")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    # tokenizer = BertTokenizer.from_pretrained(model_name_or_path)
    q_encoder = BertEncoder.from_pretrained(model_name_or_path).to(args.device)
    p_encoder = BertEncoder.from_pretrained(model_name_or_path).to(args.device)

    retriever = BiRetrieval(
        args=args,
        dataset=dataset,
        num_neg=num_neg,
        tokenizer=tokenizer,
        q_encoder=q_encoder,
        p_encoder=p_encoder,
    )

    # to save set the path
    if not model_name_or_path.startswith("../model/"):
        model_name_or_path = "../model/" + model_name_or_path
        if model_name_or_path.endswith("/"):
            model_name_or_path = model_name_or_path[:-1]

    if train:
        print("training start")
        retriever.train()
        retriever.save_pretrained(model_name_or_path)

        query = ["유아인에게 타고난 배우라고 말한 드라마 밀회의 감독은?"]
        doc_scores, doc_indices = retriever.get_relevant_doc_dot_prod(
            query=query, topk=5
        )

        for i in range(len(doc_scores)):
            print("query:", query[i])
            for j in range(len(doc_scores[i])):
                id, score = doc_indices[i][j], doc_scores[i][j]
                print(f"Top {j} passage id: {id}, score: {score}")
                pprint(retriever.dataset["context"][id])

    if evaluate:
        print("evaluation start")
        retriever.from_pretrained(model_name_or_path)
        # retriever.eval()
        retriever.q_encoder.eval()
        retriever.p_encoder.eval()

        query = [
            "유럽 초기에 천문학자가 누구에게서 관측소 자금을 후원 받고 큰 규모의 관측 기구를 만들었을까?",
            "유아인에게 타고난 배우라고 말한 드라마 밀회의 감독은?",
        ]
        doc_scores, doc_indices = retriever.get_relevant_doc_dot_prod(
            query=query, topk=5
        )

        for i in range(len(doc_scores)):
            print("query:", query[i])
            for j in range(len(doc_scores[i])):
                id, score = doc_indices[i][j], doc_scores[i][j]
                print(f"Top {j} passage id: {id}, score: {score}")
                pprint(retriever.dataset["context"][id])
