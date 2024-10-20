import logging
import sys
import os
import pickle
from typing import List, NoReturn, Optional, Tuple, Union

import numpy as np
import pandas as pd
from datasets import Dataset, concatenate_datasets, load_from_disk
from tqdm.auto import tqdm
import torch

from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    AnnSearchRequest,
    WeightedRanker,
)
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from pymilvus.model.sparse import SpladeEmbeddingFunction

# 2단계 상위 경로를 시스템 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils_sparse_retrieval import timer, hit, mrr, pooling_fn

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -    %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class LearnedSparseRetrieval:
    def __init__(self, embedding_method, embedding_model_name, contexts, ids, collection_name, bgem3_type="sparse", dense_metric_type="IP") -> NoReturn:

        self.embedding_method = embedding_method
        self.embedding_model_name = embedding_model_name
        self.contexts = contexts
        self.ids = ids 
        self.collection_name = collection_name
        self.bgem3_type = bgem3_type  # sparse, dense, hybrid
        self.dense_metric_type = dense_metric_type
        self.ef = None
        self.col = None

    def get_sparse_embedding_splade(self) -> NoReturn:
        # 저장 경로 지정: Connect to Milvus given URI
        sparse_path_name = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        connections.connect(uri=f"{sparse_path_name}/model/wiki_embedding.db")

        # 모델 지정
        self.ef = SpladeEmbeddingFunction(
            model_name=self.embedding_model_name, 
            device="cuda:0")

        # collection 조회. 없으면 생성
        col_name = self.collection_name
        if utility.has_collection(col_name):
            self.col = Collection(col_name)
            index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
            self.col.create_index("sparse_vector", index)
            self.col.load()
            logger.info(f"Collection \"{self.collection_name}\" is loaded.")
            logger.info(f"Number of entities contained: {self.col.num_entities}")
        else:
            # 데이터 스키마 지정
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
                FieldSchema(name="context", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
            ]
            schema = CollectionSchema(fields, description="Collection for SPLADE embeddings")
            self.col = Collection(col_name, schema, consistency_level="Strong")
            index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
            self.col.create_index("sparse_vector", index)
            self.col.load()

            # 임베딩 생성 
            batch_size = 4
            for i in tqdm(range(0, len(self.contexts), batch_size)):
                # 마지막 배치에서 범위를 초과하지 않도록 min 사용
                end_idx = min(i + batch_size, len(self.contexts))

                batch_ids = self.ids[i:end_idx]
                batch_contexts = self.contexts[i:end_idx]

                torch.cuda.empty_cache()
                with torch.no_grad():
                    batch_docs_embeddings = self.ef.encode_documents(batch_contexts)
                    batch_entities = []
                    for j in range(len(batch_ids)):
                        entity = {
                            "id": batch_ids[j],
                            "context": batch_contexts[j],
                            "sparse_vector": batch_docs_embeddings[[j], :],
                        }
                        batch_entities.append(entity)

                    self.col.insert(batch_entities)
            logger.info(f"Number of entities inserted: {self.col.num_entities}")

    def get_sparse_embedding_bgem3(self) -> NoReturn:
        # 저장 경로 지정: Connect to Milvus given URI
        sparse_path_name = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        connections.connect(uri=f"{sparse_path_name}/model/wiki_embedding.db")

        # 모델 지정
        self.ef = BGEM3EmbeddingFunction(
            model_name=self.embedding_model_name, # Specify the model name
            device="cuda:0", # Specify the device to use, e.g., 'cpu' or 'cuda:0'
            use_fp16=True, # Specify whether to use fp16. Set to `False` if `device` is `cpu`.
            return_sparse=True, # only allow the dense embedding output
            return_dense=True # only allow the sparse embedding output
        )
        dense_dim = self.ef.dim['dense']        

        # collection 조회. 없으면 생성
        col_name = self.collection_name
        if utility.has_collection(col_name):
            self.col = Collection(col_name)
            index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
            self.col.create_index("sparse_vector", index)
            index = {"index_type": "FLAT", "metric_type": self.dense_metric_type} 
            self.col.create_index("dense_vector", index)
            self.col.load()
            logger.info(f"Collection \"{self.collection_name}\" is loaded.")
            logger.info(f"Number of entities contained: {self.col.num_entities}")
        else:
            # 데이터 스키마 지정
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True), #auto_id=True),
                FieldSchema(name="context", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
                FieldSchema(name="dense_vector", dtype=DataType.FLOAT16_VECTOR, dim=dense_dim)
            ]
            schema = CollectionSchema(fields, description="Collection for BGE M3 embeddings")
            self.col = Collection(col_name, schema, consistency_level="Strong")
            index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
            self.col.create_index("sparse_vector", index)
            index = {"index_type": "FLAT", "metric_type": self.dense_metric_type} 
            self.col.create_index("dense_vector", index)
            self.col.load()

            # 임베딩 생성 
            batch_size = 4
            for i in tqdm(range(0, len(self.contexts), batch_size)):
                # 마지막 배치에서 범위를 초과하지 않도록 min 사용
                end_idx = min(i + batch_size, len(self.contexts))

                batch_ids = self.ids[i:end_idx]
                batch_contexts = self.contexts[i:end_idx]

                torch.cuda.empty_cache()
                with torch.no_grad():
                    batch_docs_embeddings = self.ef.encode_documents(batch_contexts)
                    batch_sparse_embeddings = batch_docs_embeddings['sparse']
                    batch_dense_embeddings = batch_docs_embeddings['dense']

                    batch_entities = []
                    for j in range(len(batch_ids)):
                        entity = {
                            "id": batch_ids[j],
                            "context": batch_contexts[j],
                            "sparse_vector": batch_sparse_embeddings[[j], :],
                            "dense_vector": batch_dense_embeddings[j]
                        }
                        batch_entities.append(entity)

                    self.col.insert(batch_entities)
            logger.info(f"Number of entities inserted:{self.col.num_entities}")

    def retrieve(
        self, query_or_dataset: Union[str, Dataset], dense_metric_type = "IP", topk = 10
    ) -> Union[Tuple[List, List], pd.DataFrame]:

        assert self.ef is not None and self.col is not None, "get_sparse_embedding() 메소드를 먼저 수행해주세요."

        if isinstance(query_or_dataset, str):
            query = [query_or_dataset]
            query_embeddings = self.ef.encode_queries(query)
            if self.embedding_method=="splade":
                search_result = self.sparse_search(query_embeddings, topk)
                doc_scores = []
                doc_contexts = []
                for i in range(len(search_result)):
                    doc_scores.append(search_result[i].score)
                    doc_contexts.append(search_result[i].get('context'))

                logger.info(f"[Search query]\n{query_or_dataset}\n")
                for i in range(topk):
                    logger.info(f"Top-{i+1} passage with score {doc_scores[i]:.4f}")
                    logger.debug(f"Passage: {doc_contexts[i]}")

                return (doc_scores, doc_contexts)
            
            elif self.embedding_method=="bge-m3":
                if self.bgem3_type == "sparse":
                    search_result = self.sparse_search(query_embeddings['sparse'], topk)
                    doc_scores = []
                    doc_contexts = []
                    for i in range(len(search_result)):
                        doc_scores.append(search_result[i].score)
                        doc_contexts.append(search_result[i].get('context'))
                    logger.info(f"[Sparse - Search query]\n{query_or_dataset}\n")
                    for i in range(topk):
                        logger.info(f"Top-{i+1} passage with score {doc_scores[i]:.4f}")
                        logger.debug(f"Passage: {doc_contexts[i]}")
                    return (doc_scores, doc_contexts)
                
                elif self.bgem3_type == "dense":
                    search_result = self.dense_search(query_embeddings['dense'], dense_metric_type, topk)
                    doc_scores = []
                    doc_contexts = []
                    for i in range(len(search_result)):
                        doc_scores.append(search_result[i].score)
                        doc_contexts.append(search_result[i].get('context'))
                    logger.info(f"[Dense - Search query]\n{query_or_dataset}\n")
                    for i in range(topk):
                        logger.info(f"Top-{i+1} passage with score {doc_scores[i]:.4f}")
                        logger.debug(f"Passage: {doc_contexts[i]}")
                    return (doc_scores, doc_contexts)

                elif self.bgem3_type == "hybrid":
                    search_result = self.hybrid_search(query_embeddings['dense'], query_embeddings['sparse'], dense_metric_type, sparse_weight=1.0, dense_weight=1.0, topk=topk)
                    doc_scores = []
                    doc_contexts = []
                    for i in range(len(search_result)):
                        doc_scores.append(search_result[i].score)
                        doc_contexts.append(search_result[i].get('context'))
                    logger.info(f"[Hybrid - Search query]\n{query_or_dataset}\n")
                    for i in range(topk):
                        logger.info(f"Top-{i+1} passage with score {doc_scores[i]:.4f}")
                        logger.debug(f"Passage: {doc_contexts[i]}")
                    return (doc_scores, doc_contexts)

        elif isinstance(query_or_dataset, Dataset):
            logger.info("Retrieving for dataset queries.")
            queries = query_or_dataset["question"]
            query_embeddings = self.ef.encode_queries(queries)
            if self.embedding_method == "splade":
                search_result = self.sparse_search(query_embeddings, topk)
            elif self.embedding_method == "bge-m3":
                if self.bgem3_type == "sparse":
                    search_result = self.sparse_search(query_embeddings['sparse'], topk)
                elif self.bgem3_type == "dense":
                    search_result = self.dense_search(query_embeddings['dense'], dense_metric_type, topk)
                elif self.bgem3_type == "hybrid":
                    search_result = self.hybrid_search(query_embeddings['dense'], query_embeddings['sparse'], dense_metric_type, sparse_weight=1.0, dense_weight=1.0, topk=topk)
            
            retrieved_data = []  # dictionary list to save result
            retrieved_contexts_list = []  
            for idx, example in enumerate(query_or_dataset):
                retrieved_contexts = [hit.get("context") for hit in search_result[idx]]
                retrieved_contexts_list.append(retrieved_contexts)

                retrieved_dict = {
                    "question": example['question'],
                    "id": example['id'],
                    "context": " ".join(retrieved_contexts)
                }

                if "context" in example.keys() and "answers" in example.keys():
                    retrieved_dict["original_context"] = example["context"]
                    retrieved_dict["answers"] = example["answers"]
                    try:
                        retrieved_dict["rank"] = (
                            retrieved_contexts.index(example["context"]) + 1
                        )
                    except ValueError:
                        retrieved_dict["rank"] = 0  # 정답 문서가 없으면 0

                retrieved_data.append(retrieved_dict)

            retrieved_df = pd.DataFrame(retrieved_data)
            logger.info("Completed retrieval for dataset queries.")
            return retrieved_df, retrieved_contexts_list


    def dense_search(self, query_dense_embedding, dense_metric_type="IP", topk=10):
        search_params = {"metric_type": dense_metric_type, "params": {}}
        res = self.col.search(
            query_dense_embedding,
            anns_field="dense_vector",
            limit=topk,
            output_fields=["context"],
            param=search_params,
        )
        return res
    
    def sparse_search(self, query_sparse_embedding, topk=10):
        search_params = {"metric_type": "IP","params": {}}
        res = self.col.search(
            query_sparse_embedding,
            anns_field="sparse_vector",
            limit=topk,
            output_fields=["context"],
            param=search_params,
        )
        return res
    
    def hybrid_search(
            self,
            query_dense_embedding, query_sparse_embedding,
            dense_metric_type='IP',
            sparse_weight=1.0, dense_weight=1.0,
            topk=10,
    ):

        dense_search_params = {"metric_type": dense_metric_type, "params": {}}
        dense_req = AnnSearchRequest(
            query_dense_embedding, "dense_vector", dense_search_params, limit=topk
        )
        sparse_search_params = {"metric_type": "IP", "params": {}}
        sparse_req = AnnSearchRequest(
            query_sparse_embedding, "sparse_vector", sparse_search_params, limit=topk
        )
        rerank = WeightedRanker(sparse_weight, dense_weight)
        res = self.col.hybrid_search(
            [sparse_req, dense_req], rerank=rerank, limit=topk, output_fields=["context"]
        )
        return res


    def evaluate(
        self,
        retrieved_df,
        topk,
        eval_metric: str = "hit",
    ):

        if eval_metric == "hit":  # k개의 추천 중 선호 아이템이 있는지 측정
            hit_at_k = hit(retrieved_df)
            logger.info(f"Hit@{topk}: {hit_at_k:.4f}")
            return hit_at_k

        elif eval_metric == "mrr":
            mrr_at_k = mrr(retrieved_df)
            logger.info(f"MRR@{topk}: {mrr_at_k:.4f}")
            return mrr_at_k

        else:
            logger.warning("Unsupported evaluation method: %s", eval_metric)
            return None
