import json
import pandas as pd
import numpy as np
import sys
import os

# 절대 경로로 'utils' 폴더를 sys.path에 추가
current_file_path = os.path.abspath(__file__)
utils_path = os.path.abspath(
    os.path.join(current_file_path, "../../../retrieval/sparse/utils")
)
sparse_path = os.path.abspath(
    os.path.join(current_file_path, "../../../retrieval/sparse/src")
)
# print(f"UTILS_PATH: {utils_path}")  # 경로가 올바르게 출력되는지 확인
sys.path.append(utils_path)
sys.path.append(sparse_path)

from utils_sparse_retrieval import load_config
from sparse_retrieval import SparseRetrieval
from utils_common import df_to_dataset
from transformers import AutoTokenizer


class Sparse_Model:

    def __init__(self, args, args_inf, test_datasets, wiki_path, valid_dataset):
        self.args = args
        self.args_inf = args_inf
        self.args_ret = load_config(args_inf.retrieval_config)
        self.wiki, self.ids = self.wiki_load(wiki_path)
        self.test_datasets = test_datasets
        self.valid_datasets = valid_dataset
        if self.args_ret.tokenizer_model_name == "mecab":
            from konlpy.tag import Mecab

            self.tokenizer = Mecab().morphs
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.args_ret.tokenizer_model_name, use_fast=True
            )
            self.tokenizer = self.tokenizer.tokenize

        self.retrieval = SparseRetrieval(
            embedding_method=self.args_ret.embedding_method,
            tokenizer=self.tokenizer,
            contexts=self.wiki,
            document_ids=self.ids,
        )

    def wiki_load(self, wiki_path):
        with open(wiki_path, "r", encoding="utf-8") as f:
            wiki = json.load(f)
        wiki_df = pd.DataFrame(wiki.values())
        wiki_unique_df = wiki_df.drop_duplicates(subset=["text"], keep="first")
        contexts = wiki_unique_df["text"].tolist()  # unique text 추출
        ids = wiki_unique_df["document_id"].tolist()
        return contexts, ids

    def get_contexts(self, datasets):

        # load Sparse_Embeddings
        method = self.args_ret.embedding_method

        match method:
            case "tfidf":
                self.retrieval.get_sparse_embedding_tfidf()
                pass
            case "bm25":
                self.retrieval.get_sparse_embedding_bm25()
                pass

        # datasets은 test_dataset["validation"]과 같은 구조를 가짐

        df = self.retrieval.retrieve(
            datasets, topk=self.args_ret.topk, save=False, retrieval_save_path="./"
        )
        if self.args.do_eval:
            df = df.drop(
                columns=["original_context", "rank", "document_id"], errors="ignore"
            )
            df["answers"] = df["answers"].apply(
                lambda x: {
                    "text": x["text"],
                    "answer_start": [int(start) for start in x["answer_start"]],
                }
            )

        context_dataset = df_to_dataset(df, self.args.do_predict, self.args.do_eval)

        return context_dataset
