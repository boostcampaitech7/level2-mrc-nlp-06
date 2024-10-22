import json
import pandas as pd
import numpy as np
import sys
import os
# 절대 경로로 'utils' 폴더를 sys.path에 추가
current_file_path = os.path.abspath(__file__)
utils_path = os.path.abspath(os.path.join(current_file_path, "../../../retrieval/dense/utils"))
dense_path = os.path.abspath(os.path.join(current_file_path, "../../../retrieval/dense/src"))
# print(f"UTILS_PATH: {utils_path}")  # 경로가 올바르게 출력되는지 확인
sys.path.append(utils_path)
sys.path.append(dense_path)

from utils_common import df_to_dataset, json_to_config
from transformers import AutoTokenizer

from sbert import BiEncoder, CrEncoder

class Dense_Model():
    def __init__(self, args_inf, test_datasets, wiki_path, valid_dataset):
        self.args_inf = args_inf
        self.args_ret = json_to_config(args_inf.retrieval_config)
        self.wiki = self.wiki_load(wiki_path)
        self.test_datasets = test_datasets
        self.valid_datasets = valid_dataset

        self.BiEncoder=None
        self.CrEncoder=None

    def wiki_load(self, wiki_path):
        with open(wiki_path, "r", encoding='utf-8') as f:
            corpus = json.load(f)
        print(f"loading documents\n")
        documents = [corpus[str(i)]['text'] for i in range(len(corpus))]
        return documents
    
    def get_contexts(self, datasets):
        # Only Test 
        method = self.args_ret.encoder_type
        match method:
            case "2-stages":
                bi_config = self.args_ret['bi-encoder']
                cr_config = self.args_ret['cr-encoder']

                retriever = BiEncoder(self.args_ret['data_path'], bi_config['model'])
                reranker = CrEncoder(self.args_ret['data_path'], cr_config['model'])

                hits = retriever.retrieve(data_type=self.args_ret['data_type'], stage=2, topk=bi_config['topk'])
                result_df = reranker.rerank(hits=hits, data_type=self.args_ret['data_type'], topk=cr_config['topk'], save_path='./retrieval_outputs')
            
            case "bi-encoder":
                bi_config = self.args_ret['bi-encoder']
                
                model = BiEncoder(self.args_ret['data_path'], bi_config['name'])
                model.retrive(data_type=bi_config['data_type'], stage=1, topk=bi_config['topk'], save_path="./retrieval_outputs")

            case "cr-encoder":
                cr_config = self.args_ret['cr-encoder']
                
                model = CrEncoder(self.args_ret['data_path'], cr_config['name'])
                model.retrieve(data_type=self.args_ret['data_type'], topk=cr_config['topk'], save_path='./retrieval_outputs')

        return df_to_dataset(result_df, True)