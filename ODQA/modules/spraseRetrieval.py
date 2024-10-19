import json
import pandas as pd
import numpy as np
import sys
sys.path.append('/data/ephemeral/home/jh/level2-mrc-nlp-06/')

from retrieval.sparse.utils.utils_sparse_retrieval import load_config
from retrieval.sparse.src.sparse_retrieval import SparseRetrieval
from ODQA.utils_common import df_to_dataset
from transformers import AutoTokenizer

class Sparse_Model():

    def __init__(self, args_inf, test_datasets, wiki_path, valid_dataset):
        self.args_inf = args_inf
        self.args_ret = load_config(args_inf.retrieval_config)
        self.wiki = self.wiki_load(wiki_path)
        self.test_datasets = test_datasets
        self.valid_datasets = valid_dataset
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args_ret.tokenizer_model_name,
            use_fast = True
        )
        self.retrieval = SparseRetrieval(
            embedding_method=self.args_ret.embedding_method,
            tokenizer = self.tokenizer,
            contexts = self.wiki
        )

    def wiki_load(self, wiki_path):
        with open(wiki_path, "r", encoding="utf-8") as f:
            wiki = json.load(f)
        wiki_df = pd.DataFrame(wiki.values())
        wiki_unique_df = wiki_df.drop_duplicates(subset=["text"], keep="first")
        contexts = wiki_unique_df["text"].tolist()  # unique text 추출
        return contexts
    
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
        df = self.retrieval.retrieve(datasets, topk=self.args_ret.topk)
        
        context_dataset = df_to_dataset(df, True)

        return context_dataset