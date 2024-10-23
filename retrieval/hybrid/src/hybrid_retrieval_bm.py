import json
import os
from multiprocessing import Pool, cpu_count
import pickle
import time
import random
from collections import Counter
from tqdm.auto import tqdm
from contextlib import contextmanager
from typing import List, NoReturn, Optional, Tuple, Union

import faiss
import numpy as np
import pandas as pd
from datasets import Dataset, concatenate_datasets, load_from_disk
from transformers import AutoConfig, AutoTokenizer, AutoModel, AutoModelForQuestionAnswering
import torch
from torch.utils.data import DataLoader

import bm25s
from bm25s.tokenization import Tokenizer as BM25s_Tokenizer
from rank_bm25 import BM25Okapi
#from bm25_pt import BM25
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import spmatrix
import umap

from utils import *

seed = 2024
random.seed(seed) # python random seed 고정
np.random.seed(seed) # numpy random seed 고정

class BM25():
    def __init__(self, contexts=None, tokenizer=None):
        self.k1 = 1.2
        self.b = 0.75
        self.avg_dl = 0
        self.n_d = 0
        
        self.D = contexts
        self.D_V = None
        self.D_1 = None

        self.tokenizer = tokenizer
        self.max_vocab_size = 0
        self.vocab_to_idx = {}
        

        
        if self.D is not None:self.initialize()

    def initialize(self, contexts=None):
        with Pool() as pool:
            if contexts is not None:self.D = contexts

            D = self.D
            self.n_d = len(D)
            
            D_1 = pool.map(len, D)
            self.D_1= D_1 = np.array(D_1).reshape(self.n_d, 1) # [D, 1]
            self.avg_dl = D_1.mean()
            
            tokenized_D = pool.map(self.tokenizer.tokenize, D)
            set_D = pool.map(set, tokenized_D)
            vocab = set()
            for set_d in set_D:
                vocab |= set_d

            self.max_vocab_size = max_vocab_size = 50000 # like sklearn tfidfv
            vocab = sorted(vocab)
            vocab_padding_size = max(0, max_vocab_size - len(vocab))
            vocab += [0] * vocab_padding_size
            self.vocab_to_idx = vocab_to_idx = {v:i for i, v in enumerate(vocab)}

            t = pool.map(self.tf_d, D)
            self.D_V = D_V = np.array(t) # [D, V]
                
            #self.D_V = D_V = np.array(pool.map(self.tf_d, D)) 

    def tf_d(self, d):
        cnt = Counter(d)
        tf_d = [0] * self.max_vocab_size
        for word in cnt:
            if word in self.vocab_to_idx:
                tf_d[self.vocab_to_idx[word]] = cnt[word]
        return tf_d
    
    def Q_tf_D_f(self, q):
        return [[self.D_V[D_i][self.vocab_to_idx[q_v]] if q_v in self.vocab_to_idx else 0 for D_i in range(self.n_d)] for q_v in q]

    def Q_df_f(self, q):
        return [sum([int(self.D_V[D_i][self.vocab_to_idx[q_v]]>0) if q_v in self.vocab_to_idx else 0 for D_i in range(self.n_d) ]) for q_v in q]


    def get_scores(self, Q, topk=1):
        k1 = self.k1
        b = self.b
        
        n_q = len(Q)

        """
        Q_new = []
        for q in Q:
            q = self.sparse_tokenizer.tokenize(q)
            q += [0] * (512 - len(q))
            Q_new += [q]
        Q = Q_new
        """

        with Pool() as pool:
            Q = self.tokenizer(Q, padding=True, truncation=True, return_tensors='pt')["input_ids"]

            Q = pool.map(self.tokenizer.tokenize, Q)

            if isinstance(Q, torch.Tensor):
                Q = Q.detach().cpu().numpy()

            print("Q shape", Q.shape)
            Q_tf_D = pool.map(self.Q_tf_D_f, Q) # [Q, Q_L, D]?
            Q_tf_D = np.array(Q_tf_D)
            
            Q_df = pool.map(self.Q_df_f, Q)    # [Q, Q_L] # 그냥 set(q) x set(d) 해야하나
            Q_df = np.array(Q_df)
            
            Q_idf = np.log(1 + (self.n_d - Q_df + 0.5) / (Q_df + 0.5)) # [Q, Q_L]
            Q_idf = np.repeat(Q_idf, self.n_d, axis=1).reshape(n_q, Q.shape[1], -1)

            print(Q_tf_D.shape)
            print(Q_df.shape)
            print(Q_idf.shape)

        doc_scores = np.sum(Q_idf * Q_tf_D * (k1 + 1) / (Q_tf_D + k1 * (1 - b + b * self.D_1.reshape(1, -1) / self.avg_dl)), axis = 1) # [Q, D]
        doc_indices = np.argsort(doc_scores)[::-1]
        doc_scores = np.take_along_axis(doc_scores, doc_indices, axis=-1)

        doc_indices = doc_indices[:, :topk].tolist()
        doc_scores = doc_scores[:, :topk].tolist()
        
        return doc_scores, doc_indices



class HybridRetrieval_BM:
    def __init__(
        self,
        model_args=None,
        tokenize_fn=None,
        data_path: Optional[str] = "../data/",
        context_path: Optional[str] = "wikipedia_documents.json",
    ) -> NoReturn:

        """
        Arguments:
            tokenize_fn:
                기본 text를 tokenize해주는 함수입니다.
                아래와 같은 함수들을 사용할 수 있습니다.
                - lambda x: x.split(' ')
                - Huggingface Tokenizer
                - konlpy.tag의 Mecab

            data_path:
                데이터가 보관되어 있는 경로입니다.

            context_path:
                Passage들이 묶여있는 파일명입니다.

            data_path/context_path가 존재해야합니다.

        Summary:
            Passage 파일을 불러오고 TfidfVectorizer를 선언하는 기능을 합니다.
        """

        self.model_args = model_args

        self.data_path = data_path
        with open(context_path, "r", encoding="utf-8") as f:
            wiki = json.load(f)

        self.contexts = list(
            dict.fromkeys([v["text"] for v in wiki.values()])
        )  # set 은 매번 순서가 바뀌므로
        print(f"Lengths of unique contexts : {len(self.contexts)}")
        self.ids = list(range(len(self.contexts)))

        self.tokenize_fn = tokenize_fn

        # Transform by vectorizer
        self.tfidfv = None
        self.dimension_reducer = None
        self.bm25 = None

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.q_encoder = None
        self.p_encoder = None
        self.model = None
        self.dense_tokenizer = None
        self.sparse_tokenizer = None
        self.dense_tokenizer = None
        
        self.p_sparse_embedding = None  # get_passage_sparse_embedding()로 생성합니다
        self.p_dense_embedding = None  # get_passage_dense_embedding()로 생성합니다
        self.p_hybrid_embedding = None # get_passage_hybrid_embedding()로 생성합니다
        self.indexer = None  # build_faiss()로 생성합니다.

    def set_sparse_tokenizer(self, tokenizer):
        self.sparse_tokenizer = tokenizer

    def set_dense_tokenizer(self, tokenizer):
        self.dense_tokenizer = tokenizer

    def set_dense_encoder(self, q_encoder, p_encoder):
        self.q_encoder = q_encoder
        self.p_encoder = p_encoder

    def get_passage_sparse_embedding(self, retriever_type="tfidf") -> NoReturn:

        """
        Summary:
            Passage Embedding을 만들고
            TFIDF와 Embedding을 pickle로 저장합니다.
            만약 미리 저장된 파일이 있으면 저장된 pickle을 불러옵니다.
        """

        model_name = self.model_args.model_name_or_path

        self.sparse_tokenizer = AutoTokenizer.from_pretrained(model_name) # use_fast=False

        # Transform by vectorizer
        self.tfidfv = TfidfVectorizer(
            tokenizer=self.sparse_tokenizer.tokenize, ngram_range=(1, 2), max_features=50000,
        )

        # Pickle을 저장합니다.
        pickle_name = f"{model_name.replace('/', '_').replace('//', '_')}_{retriever_type}_embedding.bin"
        tfidfv_name = f"{model_name.replace('/', '_').replace('//', '_')}_{retriever_type}_v.bin"

        emb_path = os.path.join(self.data_path, pickle_name)
        tfidfv_path = os.path.join(self.data_path, tfidfv_name)

        if os.path.isfile(emb_path) and os.path.isfile(tfidfv_path):
            self.p_sparse_embedding = load_pickle(emb_path)
            self.tfidfv = load_pickle(tfidfv_path)
            print("Embedding pickle load.")
        else:
            if retriever_type == "tfidf":
                with timer("Build passage embedding"):
                    self.p_sparse_embedding = self.tfidfv.fit_transform(self.contexts)
                    print("passage sparse embedding shape:", self.p_sparse_embedding.shape)
                
                save_pickle(self.p_sparse_embedding, emb_path)
                save_pickle(self.tfidfv, tfidfv_path)
                print("Embedding pickle saved.")
            elif retriever_type == "bm25":
                with timer("Build passage embedding"):
                    #tokenized_corpus = [self.sparse_tokenizer.tokenize(context) for context in self.contexts]
                    self.bm25 = BM25(tokenizer=self.sparse_tokenizer)
                    self.bm25.index(self.contexts)

                #save_pickle(self.p_sparse_embedding, emb_path)
                #save_pickle(self.tfidfv, tfidfv_path)
                #print("Embedding pickle saved.")
                
    
        return self.p_sparse_embedding
    
    def get_passage_dense_embedding(self) -> NoReturn:

        """
        Summary:
            Passage Embedding을 만들고
            Embedding을 pickle로 저장합니다.
            만약 미리 저장된 파일이 있으면 저장된 pickle을 불러옵니다.
        """

        model_args = self.model_args

        # Pickle을 저장합니다.
        model_name = model_args.model_name_or_path
        embedding_name = f"{model_name.replace('/', '_').replace('//', '_')}_bi_dense_embedding.bin"

        emb_path = os.path.join(self.data_path, embedding_name)
        model_path = os.path.join(self.data_path, model_name)
                
        print("model_path",model_path)

        if os.path.isfile(emb_path):# and os.path.isdir(model_path):
            self.p_dense_embedding = load_pickle(emb_path)
            print("Passage Embedding pickle load.")

            p_encoder = self.p_encoder
            tokenizer = self.dense_tokenizer
            print("Embedding model load.")
        
            device = self.device
            dataloader = DataLoader(self.contexts, batch_size=32)
            
            p_dense_embedding_batches = []#torch.empty()
            p_dense_embedding = None

            
            # 모델의 상태 확인
            print(f"모델은 현재 {p_encoder.training} 상태입니다.") 
            print(type(p_encoder))
            
            p_encoder.to(device)
            with timer("passage dense embedding time:"):
                with torch.no_grad():
                    for i, batch in enumerate(dataloader):
                        tokenized_contexts = tokenizer(batch, padding="max_length", truncation=True, return_tensors='pt')
                        tokenized_contexts = tokenized_contexts.to(device)
                        
                        output = p_encoder(**tokenized_contexts)
                        if hasattr(output, "pooler_output"):
                            p_dense_embedding = output.pooler_output
                        elif hasattr(output, "hidden_states"):
                            p_dense_embedding = output.hidden_states[-1][:, 0, :]
                        elif hasattr(output, "last_hidden_state"):
                            p_dense_embedding = output.last_hidden_state[:, 0, :]
                        #if i == 3:
                        #    print(p_dense_embedding.shape)
                        #    break

                        #p_dense_embedding_batches = torch.cat([p_dense_embedding_batches, p_dense_embedding], dim=0)
                        p_dense_embedding_batches += [p_dense_embedding.detach().cpu()]
                    
            self.p_dense_embedding = torch.cat(p_dense_embedding_batches)


            save_pickle(self.p_dense_embedding, emb_path)
            print("Passage Embedding pickle saved.")
            
            
            if model_args.model_name_or_path.startswith("./models"):
               pass
            else: 
                #model.save_pretrained(model_path)
                #tokenizer.save_pretrained(model_path)
                #tokenizer.save_vocabulary(model_path)
                pass
        
        print("Passage dense embedding shape:", self.p_dense_embedding.shape)
        print("Passage dense embedding done.")

        return self.p_dense_embedding
    

    def get_passage_dense_embedding_cross(self) -> NoReturn:

        """
        Summary:
            Passage Embedding을 만들고
            Embedding을 pickle로 저장합니다.
            만약 미리 저장된 파일이 있으면 저장된 pickle을 불러옵니다.
        """

        model_args = self.model_args

        # Pickle을 저장합니다.
        model_name = model_args.model_name_or_path
        embedding_name = f"{model_name.replace('/', '_').replace('//', '_')}_dense_embedding.bin"

        emb_path = os.path.join(self.data_path, embedding_name)
        model_path = os.path.join(self.data_path, model_name)

        if model_args.model_name_or_path.startswith("./models"):
            model_path = model_args.model_name_or_path
        else:
            if os.path.isfile(emb_path) and os.path.isdir(model_path):
                model_args.model_name_or_path = model_path
            else:
                model_args.model_name_or_path = model_path = model_name
                

        print("model_path",model_path)

      
        if os.path.isfile(emb_path):# and os.path.isdir(model_path):
            self.p_dense_embedding = load_pickle(emb_path)
            print("Passage Embedding pickle load.")

            if model_args.model_name_or_path.startswith("./models"):
                self.model = AutoModelForQuestionAnswering.from_pretrained(model_path)
            else:
                self.model = AutoModel.from_pretrained(model_path)

            self.dense_tokenizer = AutoTokenizer.from_pretrained(model_path)
            print("Embedding model load.")
        else:
            # AutoConfig를 이용하여 pretrained model 과 tokenizer를 불러옵니다.
            # argument로 원하는 모델 이름을 설정하면 옵션을 바꿀 수 있습니다.
            config = AutoConfig.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)

            if model_args.model_name_or_path.startswith("./models"):
                model = AutoModelForQuestionAnswering.from_pretrained(
                    model_path,
                    from_tf=bool(".ckpt" in model_path),
                    config=config,
                )
            else:
                model = AutoModel.from_pretrained(
                    model_path,
                    from_tf=bool(".ckpt" in model_path),
                    config=config,
                )

            self.model = model
            self.dense_tokenizer = tokenizer

            print(f"Embedding model load.")
            device = self.device
            dataloader = DataLoader(self.contexts, batch_size=32)
            
            p_dense_embedding_batches = []#torch.empty()
            p_dense_embedding = None

            model.to(device)
            print(type(model))
            # 모델의 상태 확인
            if model.training:
                print("모델은 현재 train 상태입니다.")
            else:
                print("모델은 현재 eval 상태입니다.")

            with timer("passage dense embedding time:"):
                with torch.no_grad():
                    for i, batch in enumerate(dataloader):
                        tokenized_contexts = tokenizer(batch, padding="max_length", truncation=True, return_tensors='pt')
                        tokenized_contexts = tokenized_contexts.to(device)
                        
                        if model_args.model_name_or_path.startswith("./models"):
                            p_dense_embedding = model(**tokenized_contexts, output_hidden_states=True).hidden_states[-1][:, 0, :]
                        else:
                            output = model(**tokenized_contexts)
                            if hasattr(output, "last_hidden_state"):
                                p_dense_embedding = output.last_hidden_state[:, 0, :]
                            else:
                                p_dense_embedding = output.pooler_output
                        #if i == 3:
                        #    print(p_dense_embedding.shape)
                        #    break

                        #p_dense_embedding_batches = torch.cat([p_dense_embedding_batches, p_dense_embedding], dim=0)
                        p_dense_embedding_batches += [p_dense_embedding.detach().cpu()]
                    
            self.p_dense_embedding = torch.cat(p_dense_embedding_batches)


            save_pickle(self.p_dense_embedding, emb_path)
            print("Passage Embedding pickle saved.")
            
            
            if model_args.model_name_or_path.startswith("./models"):
               pass
            else: 
                #model.save_pretrained(model_path)
                #tokenizer.save_pretrained(model_path)
                #tokenizer.save_vocabulary(model_path)
                pass
        
        print("Passage dense embedding shape:", self.p_dense_embedding.shape)
        print("Passage dense embedding done.")

        return self.p_dense_embedding

    def get_passage_hybrid_embedding(self):

        model_name = self.model_args.model_name_or_path

        # Pickle을 저장합니다.
        hybrid_name = f"{model_name.replace('/', '_').replace('//', '_')}_hybrid_embedding.bin"
        dimension_reducer_name = f"{model_name.replace('/', '_').replace('//', '_')}_hybridv.bin"

        hybrid_path = os.path.join(self.data_path, hybrid_name)
        dimension_reducer_path = os.path.join(self.data_path, dimension_reducer_name)

        if os.path.isfile(dimension_reducer_path):
            self.dimension_reducer = load_pickle(dimension_reducer_path)
            print("Hybrid Embedding pickle load.") 

            if self.p_sparse_embedding is None:
                self.get_passage_sparse_embedding() # no return
            if self.p_dense_embedding is None:
                self.get_passage_dense_embedding() # no return

            p_sparse_embedding = self.p_sparse_embedding
            p_dense_embedding = self.p_dense_embedding

            with timer("Build passage embedding"):
                if isinstance(p_dense_embedding, torch.Tensor):
                    p_dense_embedding = p_dense_embedding.detach().to('cpu').numpy()
                else:
                    p_dense_embedding = np.array(p_dense_embedding)

                p_sparse_embedding = self.dimension_reducer.transform(p_sparse_embedding)

                a = 0.5
                #self.p_hybrid_embedding = a * self.p_sparse_embedding + (1 - a) * self.p_dense_embedding
                p_hybrid_embedding = a * p_sparse_embedding + (1 - a) * p_dense_embedding
                self.p_hybrid_embedding = p_hybrid_embedding

        else:
            if self.p_sparse_embedding is None:
                self.get_passage_sparse_embedding() # no return
            if self.p_dense_embedding is None:
                self.get_passage_dense_embedding() # no return

            p_sparse_embedding = self.p_sparse_embedding
            p_dense_embedding = self.p_dense_embedding

            with timer("Build passage embedding"):
                if isinstance(p_dense_embedding, torch.Tensor):
                    p_dense_embedding = p_dense_embedding.detach().to('cpu').numpy()
                else:
                    p_dense_embedding = np.array(p_dense_embedding)

                self.dimension_reducer = umap.UMAP(n_components=p_dense_embedding.shape[-1])
                dimension_reducer = self.dimension_reducer

                p_sparse_embedding = dimension_reducer.fit_transform(p_sparse_embedding)

                a = 0.5
                #self.p_hybrid_embedding = a * self.p_sparse_embedding + (1 - a) * self.p_dense_embedding
                p_hybrid_embedding = a * p_sparse_embedding + (1 - a) * p_dense_embedding
                self.p_hybrid_embedding = p_hybrid_embedding

            print(self.p_hybrid_embedding.shape)

            #save_pickle(dimension_reducer, dimension_reducer_path)
            #print("Hybrid Embedding pickle saved.")
        return self.p_hybrid_embedding

    def get_passage_hybrid_embedding_(self):

        model_name = self.model_args.model_name_or_path

        # Pickle을 저장합니다.
        hybrid_name = f"{model_name.replace('/', '_').replace('//', '_')}_hybrid_embedding.bin"
        dimension_reducer_name = f"{model_name.replace('/', '_').replace('//', '_')}_hybridv.bin"

        hybrid_path = os.path.join(self.data_path, hybrid_name)
        dimension_reducer_path = os.path.join(self.data_path, dimension_reducer_name)

        if os.path.isfile(hybrid_path) and os.path.isfile(dimension_reducer_path):
            self.p_hybrid_embedding = load_pickle(hybrid_path)
            self.dimension_reducer = load_pickle(dimension_reducer_path)
            print("Hybrid Embedding pickle load.") 
        else:
            if self.p_sparse_embedding is None:
                self.get_passage_sparse_embedding() # no return
            if self.p_dense_embedding is None:
                self.get_passage_dense_embedding() # no return

            p_sparse_embedding = self.p_sparse_embedding
            p_dense_embedding = self.p_dense_embedding

            with timer("Build passage embedding"):
                if isinstance(p_dense_embedding, torch.Tensor):
                    p_dense_embedding = p_dense_embedding.detach().to('cpu').numpy()
                else:
                    p_dense_embedding = np.array(p_dense_embedding)

                self.dimension_reducer = umap.UMAP(n_components=p_dense_embedding.shape[-1])
                dimension_reducer = self.dimension_reducer

                p_sparse_embedding = dimension_reducer.fit_transform(p_sparse_embedding)

                a = 0.5
                #self.p_hybrid_embedding = a * self.p_sparse_embedding + (1 - a) * self.p_dense_embedding
                p_hybrid_embedding = a * p_sparse_embedding + (1 - a) * p_dense_embedding
                self.p_hybrid_embedding = p_hybrid_embedding

            print(self.p_hybrid_embedding.shape)

            save_pickle(p_hybrid_embedding, hybrid_path)
            save_pickle(dimension_reducer, dimension_reducer_path)
            print("Hybrid Embedding pickle saved.")
        
        return self.p_hybrid_embedding

    def build_faiss(self, num_clusters = 64) -> NoReturn:

        """
        Summary:
            속성으로 저장되어 있는 Passage Embedding을
            Faiss indexer에 fitting 시켜놓습니다.
            이렇게 저장된 indexer는 `get_relevant_doc`에서 유사도를 계산하는데 사용됩니다.

        Note:
            Faiss는 Build하는데 시간이 오래 걸리기 때문에,
            매번 새롭게 build하는 것은 비효율적입니다.
            그렇기 때문에 build된 index 파일을 저정하고 다음에 사용할 때 불러옵니다.
            다만 이 index 파일은 용량이 1.4Gb+ 이기 때문에 여러 num_clusters로 시험해보고
            제일 적절한 것을 제외하고 모두 삭제하는 것을 권장합니다.
        """

        indexer_name = f"faiss_clusters{num_clusters}.index"
        indexer_path = os.path.join(self.data_path, indexer_name)
        if os.path.isfile(indexer_path):
            print("Load Saved Faiss Indexer.")
            self.indexer = faiss.read_index(indexer_path)

        else:
            p_emb = self.p_sparse_embedding.astype(np.float32).toarray()
            emb_dim = p_emb.shape[-1]

            num_clusters = num_clusters
            quantizer = faiss.IndexFlatL2(emb_dim)

            self.indexer = faiss.IndexIVFScalarQuantizer(
                quantizer, quantizer.d, num_clusters, faiss.METRIC_L2
            )
            self.indexer.train(p_emb)
            self.indexer.add(p_emb)
            faiss.write_index(self.indexer, indexer_path)
            print("Faiss Indexer Saved.")

    def retrieve(
            self, retriever_type = "two_stage", query_or_dataset: Union[str, Dataset] = None, topk: Optional[int] = 1
        ) -> Union[Tuple[List, List], pd.DataFrame]:

            """
            Arguments:
                query_or_dataset (Union[str, Dataset]):
                    str이나 Dataset으로 이루어진 Query를 받습니다.
                    str 형태인 하나의 query만 받으면 `get_relevant_doc`을 통해 유사도를 구합니다.
                    Dataset 형태는 query를 포함한 HF.Dataset을 받습니다.
                    이 경우 `get_relevant_doc_bulk`를 통해 유사도를 구합니다.
                topk (Optional[int], optional): Defaults to 1.
                    상위 몇 개의 passage를 사용할 것인지 지정합니다.

            Returns:
                1개의 Query를 받는 경우  -> Tuple(List, List)
                다수의 Query를 받는 경우 -> pd.DataFrame: [description]

            Note:
                다수의 Query를 받는 경우,
                    Ground Truth가 있는 Query (train/valid) -> 기존 Ground Truth Passage를 같이 반환합니다.
                    Ground Truth가 없는 Query (test) -> Retrieval한 Passage만 반환합니다.
            """


            if isinstance(query_or_dataset, str):
                query_or_dataset = Dataset.from_dict({"question": [query_or_dataset], "id": [0]})

            if retriever_type not in ["bm25"] and self.p_sparse_embedding is None:
                self.get_passage_sparse_embedding()
                assert self.p_sparse_embedding is not None, "get_passage_sparse_embedding() 메소드를 먼저 수행해줘야합니다."
            
            if (retriever_type not in ["bm25", "sparse"]) and self.p_dense_embedding is None:
                self.get_passage_dense_embedding()
                assert self.p_dense_embedding is not None, "get_passage_dense_embedding() 메소드를 먼저 수행해줘야합니다."

            
            if retriever_type == "hybrid" and self.p_hybrid_embedding is None:
                self.get_passage_hybrid_embedding()
                assert self.p_hybrid_embedding is not None, "get_passage_hybrid_embedding() 메소드를 먼저 수행해줘야합니다."
            
            if "bm25s" in retriever_type and self.bm25 is None:
                bm25 = bm25s.BM25()
                self.bm25 = bm25
                tokenized_corpus = bm25s.tokenize(self.contexts)
                bm25.index(tokenized_corpus)
            elif "bm25" in retriever_type and self.bm25 is None:
                ###Rank_bm25
                if self.sparse_tokenizer is None:
                    tokenizer_path = self.model_args.model_name_or_path
                    if tokenizer_path.startswith("../model/"):tokenizer_path = tokenizer_path + "/tokenizer/"
                    self.sparse_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
                with Pool() as pool:
                    #tokenized_corpus = [self.sparse_tokenizer.tokenize(context) for context in self.contexts]
                    tokenized_corpus = pool.map(self.sparse_tokenizer.tokenize, self.contexts)
                self.bm25 = BM25Okapi(tokenized_corpus)
                
                ### bm25_pt
                #self.bm25 = BM25(tokenizer=self.sparse_tokenizer, device = self.device)
                #self.bm25.index(self.contexts)

                ### custom bm25
                #if self.sparse_tokenizer is None:
                #    tokenizer_path = self.model_args.model_name_or_path
                #    if tokenizer_path.startswith("../model/"):tokenizer_path = tokenizer_path + "/tokenizer/"
                #    self.sparse_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
                #if self.bm25 is None:
                #    self.bm25 = BM25(self.contexts, self.sparse_tokenizer)
                    


            
            stage1_topk=topk*10
            stage2_topk=topk

            print("[Search queries]\n", ', '.join(query_or_dataset["question"][:3]), ", ... etc. \n")

            if retriever_type in ["bm25", "sparse", "dense", "hybrid", "two_stage", "bm25s_two_stage", "bm25s_hybrid"]:
                print(f"[Search all queries by {retriever_type}] \n")

                with timer("query exhaustive search"):
                    dataloader = DataLoader(query_or_dataset["question"], batch_size=32)
                    doc_scores_batches = []
                    doc_indices_batches = []
                    
                    for i, queries in enumerate(dataloader):
                        if "bm25s" == retriever_type:
                            tokenized_queries = bm25s.tokenize(queries)
                            doc_indices, doc_scores = self.bm25.retrieve(tokenized_queries, self.contexts, k=stage2_topk)

                            doc_indices = doc_indices.tolist()
                            doc_scores = doc_scores.tolist()

                            doc_indices = [[self.contexts.index(c) for c in doc_indices[i]] for i in range(len(doc_indices))]
                        elif "bm25" == retriever_type:
                            ### custom bm25                    
                            #doc_scores, doc_indices = self.bm25.get_scores(queries, topk=stage2_topk)

                            with Pool() as pool:
                                tokenized_queries = pool.map(self.sparse_tokenizer.tokenize, queries)
                                doc_scores = pool.map(self.bm25.get_scores, tokenized_queries)

                                doc_scores = np.array(doc_scores)
                                doc_indices = np.argsort(doc_scores)[:, ::-1]
                                doc_scores = np.take_along_axis(doc_scores, doc_indices, axis=-1)
                                
                                doc_indices = doc_indices[:, :stage2_topk].tolist()
                                doc_scores = doc_scores[:, :stage2_topk].tolist()

                            """
                            Q = query_or_dataset["question"]

                            Q_tf_D = [] # [Q, Q_L, D]?
                            for q in Q:
                                q = self.sparse_tokenizer.tokenize(q)
                                Q_tf_D += [[D_V[D_i][vocab_to_idx[q_v]] for D_i in range(n_d)] for q_v in q]

                            Q_df = [] # [Q, Q_L] # 그냥 set(q) x set(d) 해야하나
                            for q in Q:
                                Q_df += [[sum([int(tf_d[D_i][vocab_to_idx[q_v]]>0) for D_i in D]) for q_v in q]] 
                            
                            Q_idf = np.log(1 + (n_d - Q_df + 0.5) / Q_df + 0.5) # [Q, Q_L]
                            Q_idf = np.tile(Q_idf, reps=(1, 1, n_d))

                            doc_scores = np.sum(Q_idf * Q_tf_D * (k1 + 1) / (Q_tf_D + k1 * (1 - b + b * D_1 / avg_dl)), dim = 1) # [Q, D]
                            """

                            """
                            ### Rank_bm25
                            doc_scores_all = []
                            doc_indices_all = []

                            for query in queries:
                                tokenized_query = self.sparse_tokenizer.tokenize(query)
                                doc_scores = self.bm25.get_scores(tokenized_query)
                                doc_indices = self.bm25.get_top_n(tokenized_query, self.contexts, n=stage2_topk)

                                doc_scores_all += [doc_scores]
                                doc_indices_all += [doc_indices]
                            
                            doc_scores = doc_scores_all
                            doc_indices = doc_indices_all


                            doc_indices = [[self.contexts.index(c) for c in doc_indices[i]] for i in range(len(doc_indices))]
                            """

                            """
                            ### bm25_pt
                            doc_scores = self.bm25.score_batch(queries)
                            doc_indices = torch.argsort(doc_scores, descending=True)
                            doc_scores = torch.take_along_dim(doc_scores, doc_indices, dim=-1)

                            doc_indices = doc_indices.detach().cpu().numpy().tolist()
                            doc_scores = doc_indices.detach().cpu().numpy().tolist()
                            """

                        elif "two_stage" in retriever_type:
                            if "bm25s" in retriever_type:
                                tokenized_queries = bm25s.tokenize(queries)
                                doc_indices, doc_scores = self.bm25.retrieve(tokenized_queries, self.contexts, k=stage1_topk)

                                doc_indices = doc_indices.tolist()
                                doc_scores = doc_scores.tolist()

                                doc_indices = [[self.contexts.index(c) for c in doc_indices[i]] for i in range(len(doc_indices))]
                            else:
                                doc_scores, doc_indices = self.get_relevant_doc(retriever_type="sparse", queries=queries, k=stage1_topk)
                            
                            #doc_scores, doc_indices = self.get_relevant_doc(retriever_type="dense", queries=queries, doc_scores=doc_scores, doc_indices=doc_indices, topk=stage2_topk)
                            doc_scores, doc_indices = self.rerank_doc(retriever_type="dense", queries=queries, doc_scores=doc_scores, doc_indices=doc_indices, topk=stage2_topk)
                        
                        else:
                            doc_scores, doc_indices = self.get_relevant_doc(retriever_type=retriever_type, queries=queries, k=stage2_topk)

                        #print("rerank score",doc_scores)
                        doc_scores_batches += doc_scores
                        doc_indices_batches += doc_indices

                        #if i == 0:break
                    
                    doc_scores = doc_scores_batches
                    doc_indices = doc_indices_batches
                
                    if len(doc_scores) == 1:
                        for i in range(topk):
                            print(f"Top-{i+1} passage with score {doc_scores[0][i]:4f}")
                            print(self.contexts[doc_indices[0][i]][:100])
            
                    else:
                        total = []
                        for idx, example in enumerate(
                            tqdm(query_or_dataset, desc="Sparse retrieval: ")
                        ):
                            if idx >= len(doc_indices):break

                            docs = [self.contexts[pid] for pid in doc_indices[idx]]
                        
                            tmp = {
                                # Query와 해당 id를 반환합니다.
                                "question": example["question"],
                                "id": example["id"],
                                # Retrieve한 Passage의 id, context를 반환합니다.
                                "context": " ".join(docs),
                                "original_document_rank":docs.index(example["context"])+1 if example["context"] in docs else float('inf')
                            }
                            if "context" in example.keys() and "answers" in example.keys():
                                # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                                tmp["original_context"] = example["context"]
                                tmp["answers"] = example["answers"]
                            total.append(tmp)

                        cqas = pd.DataFrame(total)
                        return cqas
            
    def retrieve_sparse(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:

        """
        Arguments:
            query_or_dataset (Union[str, Dataset]):
                str이나 Dataset으로 이루어진 Query를 받습니다.
                str 형태인 하나의 query만 받으면 `get_relevant_doc`을 통해 유사도를 구합니다.
                Dataset 형태는 query를 포함한 HF.Dataset을 받습니다.
                이 경우 `get_relevant_doc_bulk`를 통해 유사도를 구합니다.
            topk (Optional[int], optional): Defaults to 1.
                상위 몇 개의 passage를 사용할 것인지 지정합니다.

        Returns:
            1개의 Query를 받는 경우  -> Tuple(List, List)
            다수의 Query를 받는 경우 -> pd.DataFrame: [description]

        Note:
            다수의 Query를 받는 경우,
                Ground Truth가 있는 Query (train/valid) -> 기존 Ground Truth Passage를 같이 반환합니다.
                Ground Truth가 없는 Query (test) -> Retrieval한 Passage만 반환합니다.
        """

        assert self.p_sparse_embedding is not None, "get_passage_sparse_embedding() 메소드를 먼저 수행해줘야합니다."

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc_sparse(query_or_dataset, k=topk)
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print(f"Top-{i+1} passage with score {doc_scores[i]:4f}")
                print(self.contexts[doc_indices[i]][:100])

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):

            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            total = []
            #with timer("query exhaustive search"):
            doc_scores, doc_indices = self.get_relevant_doc_bulk_sparse(
                query_or_dataset["question"], k=topk
            )
            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="Sparse retrieval: ")
            ):
                

                docs = [self.contexts[pid] for pid in doc_indices[idx]]
            
                tmp = {
                    # Query와 해당 id를 반환합니다.
                    "question": example["question"],
                    "id": example["id"],
                    # Retrieve한 Passage의 id, context를 반환합니다.
                    "context": " ".join(docs),
                    "original_document_rank":docs.index(example["context"])+1 if example["context"] in docs else float('inf')
                }
                if "context" in example.keys() and "answers" in example.keys():
                    # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            cqas = pd.DataFrame(total)
            return cqas
        
    def get_relevant_doc(self, retriever_type = "sparse", queries: List[str] = None, k: Optional[int] = 1) -> Tuple[List, List]:
        """
        Arguments:
            queries (List):
                하나 이상의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """

        #print("get_relevant_doc:", retriever_type)

        #with timer("queries embedding"):
        sparse_queries_vec = self.tfidfv.transform(queries)

        assert (np.sum(sparse_queries_vec) != 0), "Error: The Words of questions not in the vocab of vectorizer"
        
        #if retriever_type == "two_stage":   
        if retriever_type in ["dense", "hybrid"]:
            self.model.to(self.device)

            with torch.no_grad():
                tokenized_queries = self.dense_tokenizer(queries, padding="max_length", truncation=True, return_tensors='pt').to(self.device)

                if self.model_args.model_name_or_path.startswith("./models"):
                    dense_queries_vec = self.model(**tokenized_queries, output_hidden_states=True).hidden_states[-1][:, 0, :]
                else:
                    output = self.model(**tokenized_queries)
                    if hasattr(output, "last_hidden_state"):
                        dense_queries_vec = output.last_hidden_state[:, 0, :]
                    else:
                        dense_queries_vec = output.pooler_output

                dense_queries_vec = dense_queries_vec.detach().cpu().numpy()
                
                assert (np.sum(dense_queries_vec) != 0), "Error: The Words of questions not in the vocab of vectorizer"

        if retriever_type in ["hybrid"]:
            sparse_queries_vec = self.dimension_reducer.transform(sparse_queries_vec)

            a = 0.5
            hybrid_queries_vec = a * sparse_queries_vec + (1 - a) * dense_queries_vec
            
            print("sparse", sparse_queries_vec.shape, type(sparse_queries_vec))
            print("dense", dense_queries_vec.shape, type(dense_queries_vec))
            print("hybrid", hybrid_queries_vec.shape, type(hybrid_queries_vec))

        q_embedding = None
        p_embedding = None
        
        match retriever_type:
            case "sparse":
                q_embedding = sparse_queries_vec
                p_embedding = self.p_sparse_embedding
            case "dense":
                q_embedding = dense_queries_vec
                p_embedding = self.p_dense_embedding
            case "hybrid":
                q_embedding = hybrid_queries_vec
                p_embedding = self.p_hybrid_embedding

        #with timer("query ex search"):
        if not isinstance(q_embedding, spmatrix) or not isinstance(p_embedding, spmatrix):
            if hasattr(q_embedding, "toarray"):
                q_embedding = q_embedding.toarray()
            if hasattr(p_embedding, "toarray"):
                p_embedding = p_embedding.toarray()
            
            q_embedding = torch.Tensor(q_embedding).to(self.device)
            p_embedding = torch.Tensor(p_embedding).to(self.device)

        doc_scores = q_embedding @ p_embedding.T

        #print("doc_scores", doc_scores.shape, type(doc_scores))

        if isinstance(doc_scores, torch.Tensor):
            doc_scores = doc_scores.detach().cpu().numpy()
        if hasattr(doc_scores, "toarray"):
            doc_scores = doc_scores.toarray()

        #print("new doc_scores shape", doc_scores.shape)

        """
        doc_scores = []
        doc_indices = []
        for i in range(result.shape[0]):
            sorted_result = np.argsort(result[i, :])[::-1]
            doc_scores.append(result[i, :][sorted_result].tolist()[:k])
            doc_indices.append(sorted_result.tolist()[:k])
        
            #print("sorted_result", sorted_result)
        """

        #print("doc_scores-------")
        #doc_scores = doc_scores.squeeze(1)
        #print("doc_scores.shape:", doc_scores.shape)

        doc_indices = np.argsort(doc_scores)[:, ::-1].copy()
        doc_scores = np.take_along_axis(doc_scores, doc_indices, axis=-1)

        #print("doc score", doc_scores[:32])
        #print("doc indices", doc_indices[:32])

        return doc_scores[:, :k].tolist(), doc_indices[:, :k].tolist()
    
    
 
    def get_relevant_doc_sparse(self, queries: List[str], k: Optional[int] = 1) -> Tuple[List, List]:

        """
        Arguments:
            query (str):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """

        if type(queries) is not list:
            queries = [queries]

        with timer("transform"):
            queries_vec = self.tfidfv.transform(queries)
            #print("query_vec shape:", query_vec)
        
        assert (
            np.sum(queries_vec) != 0
        ), "Error: The Words of questions not in the vocab of vectorizer"

        with timer("queries ex search"):
            result = queries_vec * self.p_sparse_embedding.T
            #print("result shape:", result.shape)
        
        if not isinstance(result, np.ndarray):
            result = result.toarray()

        p_embedding = self.p_sparse_embedding
        #print("get_relevant_doc", "sparse")
        #print("query_vec:", type(query_vec))
        #print("query_vec.shape:", query_vec.shape)
        #print("query_vec", query_vec)
        #print("p_emb", type(p_embedding))
        #print("p_emb.shape", p_embedding.shape)
        #print("p_emb", p_embedding)
        #print("result:", type(result))
        #print("result.shape:", result.shape)

        #sorted_result = np.argsort(result.squeeze())[::-1]
        #doc_score = result.squeeze()[sorted_result].tolist()[:k]
        #doc_indices = sorted_result.tolist()[:k]

        result = result.squeeze()
        sorted_result_indices = np.argsort(result)[::-1]
        doc_score = result[sorted_result_indices][:k].tolist()
        doc_indices = sorted_result_indices[:k].tolist()

        #print("result score", doc_score[:32])
        #print("result indices", doc_indices[:32])

        return doc_score, doc_indices

    def get_relevant_doc_bulk_sparse(
        self, queries: List, k: Optional[int] = 1
    ) -> Tuple[List, List]:

        """
        Arguments:
            queries (List):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """

        query_vec = self.tfidfv.transform(queries)
        assert (
            np.sum(query_vec) != 0
        ), "Error: The Words of questions not in the vocab of vectorizer"

        result = query_vec * self.p_sparse_embedding.T

        p_embedding = self.p_sparse_embedding
        #print("get_relevant_doc_bulk", "sparse")
        #print("query_vec:", type(query_vec))
        #print("query_vec.shape:", query_vec.shape)
        #print("query_vec", query_vec)
        #print("p_emb", type(p_embedding))
        #print("p_emb.shape", p_embedding.shape)
        #print("p_emb", p_embedding)
        #print("result:", type(result))
        #print("result.shape:", result.shape)

        if not isinstance(result, np.ndarray):
            result = result.toarray()

        
        print("new result shape", result.shape)
        doc_scores = []
        doc_indices = []
        for i in range(result.shape[0]):
            sorted_result = np.argsort(result[i, :])[::-1]
            doc_scores.append(result[i, :][sorted_result].tolist()[:k])
            doc_indices.append(sorted_result.tolist()[:k])
            
            #print("sorted_result", sorted_result)

        #print("result score", doc_scores[:32])
        #print("result indices", doc_indices[:32])
        return doc_scores, doc_indices
    
    def rerank_doc(self, retriever_type = "sparse", queries = None, doc_scores = None, doc_indices = None, topk = 1):
        """
        def csr_to_coo(x_csr):
            x_coo = x_csr.tocoo()
            values = torch.tensor(x_coo.data, dtype=torch.float32)
            indices = torch.tensor([x_coo.row, x_coo.col], dtype=torch.int64)
            x_coo_tensor = torch.sparse_coo_tensor(indices, values, x_coo.shape)
            return x_coo_tensor



        p_embedding = None
        
        if retriever_type == "sparse":
            with timer("sparse reranker"):
                query_vec = self.tfidfv.transform(queries)
                p_embedding = self.p_sparse_embedding

                query_vec = csr_to_coo(query_vec)
                p_embedding = csr_to_coo(p_embedding)

                p_embedding = None
        """
        
        if type(queries) is str:queries = [queries]

        if retriever_type == "sparse":
            with timer("sparse reranker"):
                query_vec = self.tfidfv.transform(queries)
                p_embedding = self.p_sparse_embedding

                query_vec = query_vec.toarray()
                p_embedding = p_embedding.toarray()

        elif retriever_type == "dense":
            with timer("dense reranker"):
                with torch.no_grad():
                    tokenized_query = self.dense_tokenizer(queries, padding="max_length", truncation=True, return_tensors='pt')
                    tokenized_query = tokenized_query.to(self.device)

                    self.model.to(self.device)
                    if self.model_args.model_name_or_path.startswith("./models"):
                        query_vec = self.model(**tokenized_query, output_hidden_states=True).hidden_states[-1][:, 0, :]
                    else:
                        output = self.model(**tokenized_query)
                        if hasattr(output, "last_hidden_state"):
                            query_vec = output.last_hidden_state[:, 0, :]
                        else:
                            query_vec = output.pooler_output

                    query_vec = query_vec.detach().cpu().numpy()
                    #print("query_vec shape:", query_vec.shape)
                    assert (
                        np.sum(query_vec) != 0
                    ), "Error: The Words of questions not in the vocab of vectorizer"

                    p_embedding = self.p_dense_embedding

        #print(query_vec.shape)
        #print(np.array(doc_indices).shape)
        #print(p_embedding.shape)

        query_vec = torch.tensor(query_vec).to(self.device)
        p_embedding = torch.tensor(p_embedding).to(self.device)
        p_embedding = p_embedding[torch.tensor(doc_indices)]

        query_vec = query_vec.unsqueeze(1)

        #result = torch.sparse.mm(query_vec, p_embedding[doc_indices].view(query_vec.shape[0], -1, p_embedding.shape[-1]).transpose(1, 2))
        


        result = query_vec @ p_embedding.view(query_vec.shape[0], -1, p_embedding.shape[-1]).transpose(1, 2)
        
        #print("rerank_doc:", retriever_type)
        #print("query_vec:", type(query_vec))
        #print("query_vec.shape:", query_vec.shape)
        #print("query_vec", query_vec)
        #print("p_emb", type(p_embedding))
        #print("p_emb.shape", p_embedding.shape)
        #print("p_emb", p_embedding)
        #print("result:", type(result))
        #print("result.shape:", result.shape)
        #print("result:", result)
        
        #print("result shape:", result.shape)
        
        if isinstance(result, torch.Tensor):
            result = result.detach().cpu().numpy()
        elif not isinstance(result, np.ndarray):
            result = result.toarray()

        #print("result-------")
        result = result.squeeze(1)
        #print("rerank result.shape:", result.shape)

        result_sorted_indices = np.argsort(result)[:, ::-1].copy()
        #print("sorted_indices.shape", result_sorted_indices.shape)
        #print("rerank sorted indices", result_sorted_indices)
        doc_rerank_scores = np.take_along_axis(result, result_sorted_indices, axis=-1)

        doc_indices = np.array(doc_indices).reshape(result_sorted_indices.shape)
        doc_rerank_indices = np.take_along_axis(doc_indices, result_sorted_indices, axis=-1)
        

        """
        print("doc_score")
        print("doc_score.shape:", np.array(doc_scores).shape, np.array(doc_rerank_scores).shape)
        print("doc_rerank score type:", type(doc_rerank_scores))
        print("doc_rerank_score", doc_rerank_scores)
        print(doc_scores[0], "\n",doc_rerank_scores[0])
        print()
        print("doc_score")
        print("doc_indices.shape:", doc_indices.shape, np.array(doc_rerank_indices).shape)
        print("doc_rerank_indices type:", type(doc_rerank_indices))
        print("doc_rerank_indices", doc_rerank_indices)
        print(doc_indices[0], "\n", doc_rerank_indices[0])
        """

        """
        print("new result shape", result.shape)
        doc_scores_batches = []
        doc_indices_batches = []
        for i in range(result.shape[0]):
            sorted_result = np.argsort(result[i, :])[::-1]
            doc_scores_batches.append(result[i, :][sorted_result].tolist()[:topk])
            doc_indices_batches.append(sorted_result.tolist()[:topk])
            print("sorted_result", sorted_result)
        doc_rerank_score = np.array(doc_scores_batches)
        doc_rerank_indices = np.array(doc_indices_batches)

        print("rerankdocscorebatches", doc_scores_batches)
        print("rerankdocindicesbatches", doc_indices_batches)
        print("rerankdocscore", doc_rerank_score)
        print("rerankdocindices", doc_rerank_indices)
        """

        """
        doc_scores_batches = []
        doc_indices_batches = []
        for i in range(result.shape[0]):
            sorted_result = np.argsort(result[i, :])[::-1]
            doc_scores_batches.append(result[i, :][sorted_result].tolist())
            doc_indices_batches.append(sorted_result.tolist())
        doc_rerank_score = np.array(doc_scores_batches)
        doc_rerank_indices = np.array(doc_indices_batches)
        """

        return doc_rerank_scores[:, :topk].tolist(), doc_rerank_indices[:, :topk].tolist()
        
        #return doc_rerank_score, doc_rerank_indices
    
    def retrieve_faiss(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:

        """
        Arguments:
            query_or_dataset (Union[str, Dataset]):
                str이나 Dataset으로 이루어진 Query를 받습니다.
                str 형태인 하나의 query만 받으면 `get_relevant_doc`을 통해 유사도를 구합니다.
                Dataset 형태는 query를 포함한 HF.Dataset을 받습니다.
                이 경우 `get_relevant_doc_bulk`를 통해 유사도를 구합니다.
            topk (Optional[int], optional): Defaults to 1.
                상위 몇 개의 passage를 사용할 것인지 지정합니다.

        Returns:
            1개의 Query를 받는 경우  -> Tuple(List, List)
            다수의 Query를 받는 경우 -> pd.DataFrame: [description]

        Note:
            다수의 Query를 받는 경우,
                Ground Truth가 있는 Query (train/valid) -> 기존 Ground Truth Passage를 같이 반환합니다.
                Ground Truth가 없는 Query (test) -> Retrieval한 Passage만 반환합니다.
            retrieve와 같은 기능을 하지만 faiss.indexer를 사용합니다.
        """

        assert self.indexer is not None, "build_faiss()를 먼저 수행해주세요."

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc_faiss(
                query_or_dataset, k=topk
            )
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print("Top-%d passage with score %.4f" % (i + 1, doc_scores[i]))
                print(self.contexts[doc_indices[i]])

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):

            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            queries = query_or_dataset["question"]
            total = []

            with timer("query faiss search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk_faiss(
                    queries, k=topk
                )
            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="Sparse retrieval: ")
            ):
                tmp = {
                    # Query와 해당 id를 반환합니다.
                    "question": example["question"],
                    "id": example["id"],
                    # Retrieve한 Passage의 id, context를 반환합니다.
                    "context": " ".join(
                        [self.contexts[pid] for pid in doc_indices[idx]]
                    ),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            return pd.DataFrame(total)

    def get_relevant_doc_faiss(
        self, query: str, k: Optional[int] = 1
    ) -> Tuple[List, List]:

        """
        Arguments:
            query (str):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """

        query_vec = self.tfidfv.transform([query])
        assert (
            np.sum(query_vec) != 0
        ), "Error: The Words of questions not in the vocab of vectorizer"

        q_emb = query_vec.toarray().astype(np.float32)
        with timer("query faiss search"):
            D, I = self.indexer.search(q_emb, k)

        return D.tolist()[0], I.tolist()[0]

    def get_relevant_doc_bulk_faiss(
        self, queries: List, k: Optional[int] = 1
    ) -> Tuple[List, List]:

        """
        Arguments:
            queries (List):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """

        query_vecs = self.tfidfv.transform(queries)
        assert (
            np.sum(query_vecs) != 0
        ), "Error: The Words of questions not in the vocab of vectorizer"

        q_embs = query_vecs.toarray().astype(np.float32)
        D, I = self.indexer.search(q_embs, k)

        return D.tolist(), I.tolist()

    def test(self, retriever_type = "sparse", query_or_dataset = None, topk = 1):

        print(f"\n@@@@@@@@@@@@@@@@@@@@@@@@@ {retriever_type} method @@@@@@@@@@@@@@@@@@@@@@@@@") 

        with timer("exhaustive search"):
            df = self.retrieve(retriever_type=retriever_type, query_or_dataset=query_or_dataset, topk=topk)
            #df["correct"] = df["original_context"] == df["context"]
            df["hit@k"] = df.apply(lambda row:row["original_context"] in row["context"], axis=1)
            df["mrr@k"] = df.apply(lambda row:1 / row["original_document_rank"], axis=1)
            
            #print(df)
            #print(df.to_string())
            #print(
            #    "correct retrieval result by exhaustive search",
            #    df["correct"].sum() / len(df),
            #)
            print(f"hit@{topk} {retriever_type} retrieval result by exhaustive search: {df['hit@k'].sum() / len(df):.4f}")
            print(f"mrr@{topk} {retriever_type} retrieval result by exhaustive search: {df['mrr@k'].sum() / len(df):.4f}")

        with timer("An example"):
            df = self.retrieve(retriever_type=retriever_type, query_or_dataset=query_or_dataset, topk=topk)
        
        print(f"@@@@@@@@@@@@@@@@@@@@@@@@@  {retriever_type} done  @@@@@@@@@@@@@@@@@@@@@@@@@\n") 