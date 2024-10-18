import logging
import sys
import os
import pickle
from typing import List, NoReturn, Optional, Tuple, Union
# from scipy import sparse

import numpy as np
import pandas as pd
from datasets import Dataset, concatenate_datasets, load_from_disk
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForMaskedLM, AutoTokenizer

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


class SparseRetrieval:
    def __init__(self, embedding_method, contexts, 
                 tokenizer=None,
                 aggregation_method: Optional[str]=None, 
                 similarity_metric: Optional[str]=None,
                 embedding_model_name: Optional[str]=None) -> NoReturn:

        self.embedding_method = embedding_method
        self.contexts = contexts

        # 전통적인 sparse embedding (TF-IDF, BM25)
        if self.embedding_method in ["tfidf", "bm25"]:
            self.tokenize_fn = tokenizer.tokenize
            self.tfidfv = None
            self.p_embedding = None
            self.bm25 = None
        
        # Learned sparse embedding (BGE-M3)
        elif self.embedding_method == 'bge-m3':
            self.embedding_model_name = embedding_model_name
            self.aggregation_method = aggregation_method  # 'sum', 'max'
            self.similarity_metric = similarity_metric    # 'cosine', 'dot_product'
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.encoder = AutoModelForMaskedLM.from_pretrained(self.embedding_model_name)
            self.encoder.eval()
            self.encoder.to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(self.embedding_model_name)
            self.p_embedding_norm = None  # For cosine similarity



    def get_sparse_embedding_tfidf(self) -> NoReturn:
        os.makedirs("../model", exist_ok=True)
        pickle_name = f"{self.embedding_method}_sparse_embedding.bin"
        vectorizer_name = f"{self.embedding_method}.bin"
        emb_path = os.path.join("../model", pickle_name)
        vectorizer_path = os.path.join("../model", vectorizer_name)


        if os.path.isfile(vectorizer_path) and os.path.isfile(emb_path):
            logger.info("Loading TF-IDF pickle files.")
            with open(vectorizer_path, "rb") as file:
                self.tfidfv = pickle.load(file)
            with open(emb_path, "rb") as file:
                self.p_embedding = pickle.load(file)
        else:
            logger.info("Build TF-IDF passage embedding")
            self.tfidfv = TfidfVectorizer(
                tokenizer=self.tokenize_fn, ngram_range=(1, 2), max_features=50000
            )
            self.p_embedding = self.tfidfv.fit_transform(
                tqdm(self.contexts, desc="TF-IDF Vectorization")
            )

            logger.info("Saving TF-IDF pickle files.")
            with open(vectorizer_path, "wb") as file:
                pickle.dump(self.tfidfv, file)
            with open(emb_path, "wb") as file:
                pickle.dump(self.p_embedding, file)
            logger.info("Embedding pickle saved.")

    def get_sparse_embedding_bm25(self) -> NoReturn:
        os.makedirs("../model", exist_ok=True)
        vectorizer_name = f"{self.embedding_method}.bin"
        vectorizer_path = os.path.join("../model", vectorizer_name)

        if os.path.isfile(vectorizer_path):  # bm25 does not use p_embedding
            logger.info("Loading BM25 pickle file.")
            with open(vectorizer_path, "rb") as file:
                self.bm25 = pickle.load(file)
        else:
            logger.info("Fitting BM25 model.")
            tokenized_corpus = [
                self.tokenize_fn(doc)
                for doc in tqdm(self.contexts, desc="Tokenizing for BM25")
            ]
            self.bm25 = BM25Okapi(tokenized_corpus)

            logger.info("Saving BM25 pickle file.")
            with open(vectorizer_path, "wb") as file:
                pickle.dump(self.bm25, file)
            logger.info("BM25 model saved.")


    def get_sparse_embedding_learned(self, batch_size: int = 8) -> NoReturn:
        os.makedirs("../model", exist_ok=True)
        pickle_name = f"{self.embedding_method}_sparse_embedding.bin"
        emb_path = os.path.join("../model", pickle_name)
        
        if os.path.isfile(emb_path): 
            logger.info("Loading embedding pickle file.")
            with open(emb_path, "rb") as file:
                self.p_embedding = pickle.load(file)
            logger.info("Loading is done")
        else:
            logger.info(f"Generating embeddings using {self.embedding_method} model.")
            dataloader = DataLoader(self.contexts, batch_size=batch_size)
            current_batch_embeddings = []

            with timer("BGE-M3 Embedding Generation"):
                for batch in tqdm(dataloader, desc="Generating Passage Embeddings"):
                    encoded_input = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(self.device)
                    with torch.no_grad():
                        model_output = self.encoder(**encoded_input)
                        values = pooling_fn(encoded_input, model_output, self.aggregation_method)
                        batch_embedding = values.squeeze().cpu().numpy().astype(np.float16)
                        current_batch_embeddings.append(batch_embedding)
                    # GPU 캐시 삭제
                    del encoded_input, model_output, values, batch_embedding
                    torch.cuda.empty_cache()
                self.p_embedding = np.vstack(current_batch_embeddings)

            logger.info("Saving BGE-M3 pickle file.")
            with open(emb_path, "wb") as file:
                pickle.dump(self.p_embedding, file)
            logger.info("BGE-M3 model saved.")

        if self.similarity_metric == "cosine":
            print("normaliztion start")
            eps = 1e-9

            # Normalize embeddings batch by batch
            normalization_batch_size = 1000
            total_embeddings = self.p_embedding.shape[0]
            for start_idx in tqdm(range(0, total_embeddings, normalization_batch_size), desc="Normalizing embeddings"):
                end_idx = min(start_idx + normalization_batch_size, total_embeddings)
                batch_embeddings = self.p_embedding[start_idx:end_idx]
                norms = (np.linalg.norm(batch_embeddings, axis=1, keepdims=True) + eps)
                self.p_embedding[start_idx:end_idx] = batch_embeddings / norms

            # p_embedding_norm = (np.linalg.norm(self.p_embedding, axis=1, keepdims=True) + eps)
            # self.p_embedding = self.p_embedding / p_embedding_norm
            logger.info("Normalization finished.")

    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:

        if self.embedding_method in ["tfidf", "bge-m3"]:
            assert (
                self.p_embedding is not None
            ), "get_sparse_embedding() 메소드를 먼저 수행해주세요."
        elif self.embedding_method == "bm25":
            assert (
                self.bm25 is not None
            ), "get_sparse_embedding() 메소드를 먼저 수행해주세요."

        if isinstance(query_or_dataset, str):
            logger.info("Retrieving for single query.")
            if self.embedding_method == "tfidf":
                doc_scores, doc_indices = self.get_relevant_doc_tfidf(
                    query_or_dataset, k=topk
                )
            elif self.embedding_method == "bm25":
                doc_scores, doc_indices = self.get_relevant_doc_bm25(
                    query_or_dataset, k=topk
                )
            elif self.embedding_method == "bge-m3":
                doc_scores, doc_indices = self.get_relevant_doc_learned(
                    query_or_dataset, k=topk
                )
            logger.info(f"[Search query]\n{query_or_dataset}\n")

            for i in range(topk):
                logger.info(f"Top-{i+1} passage with score {doc_scores[i]:.4f}")
                logger.debug(f"Passage: {self.contexts[doc_indices[i]]}")

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):
            logger.info("Retrieving for dataset queries.")
            queries = query_or_dataset["question"]
            retrieved_data = []  # dictionary list to save result

            with timer("query exhaustive search"):
                if self.embedding_method == "tfidf":
                    doc_scores, doc_indices = self.get_relevant_doc_bulk_tfidf(
                        queries, k=topk
                    )
                elif self.embedding_method == "bm25":
                    doc_scores, doc_indices = self.get_relevant_doc_bulk_bm25(
                        queries, k=topk
                    )
                elif self.embedding_method == "bge-m3":
                    doc_scores, doc_indices = self.get_relevant_doc_bulk_learned(
                        queries, k=topk
                    )

            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="Sparse retrieval: ")
            ):
                retrieved_contexts = [self.contexts[pid] for pid in doc_indices[idx]]

                retrieved_dict = {
                    # 쿼리와 해당 ID 반환
                    "question": example["question"],
                    "id": example["id"],
                    # retrieve한 passage의 context 이어 붙이기
                    "context": " ".join(retrieved_contexts),
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

            return retrieved_df

    def get_relevant_doc_tfidf(
        self, query: str, k: Optional[int] = 1
    ) -> Tuple[List, List]:

        query_vec = self.tfidfv.transform([query])
        if query_vec.nnz == 0:
            logger.warning("Query contains only unknown words.")
            raise ValueError(
                "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."
            )

        result = query_vec * self.p_embedding.T
        if not isinstance(result, np.ndarray):
            result = result.toarray()

        sorted_result = np.argsort(result.squeeze())[::-1]
        doc_score = result.squeeze()[sorted_result].tolist()[:k]
        doc_indices = sorted_result.tolist()[:k]
        return doc_score, doc_indices

    def get_relevant_doc_bulk_tfidf(
        self, queries: List, k: Optional[int] = 1
    ) -> Tuple[List, List]:

        query_vecs = self.tfidfv.transform(queries)
        if query_vecs.nnz == 0:
            logger.warning("One or more queries contain only unknown words.")
            raise ValueError(
                "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."
            )

        result = query_vecs * self.p_embedding.T
        if not isinstance(result, np.ndarray):
            result = result.toarray()
        doc_scores = []
        doc_indices = []
        for i in range(result.shape[0]):
            sorted_result = np.argsort(result[i, :])[::-1]
            doc_scores.append(result[i, :][sorted_result].tolist()[:k])
            doc_indices.append(sorted_result.tolist()[:k])
        return doc_scores, doc_indices

    def get_relevant_doc_bm25(
        self, query: str, k: Optional[int] = 1
    ) -> Tuple[List, List]:
        tokenized_query = self.tokenize_fn(query)
        doc_scores = self.bm25.get_scores(tokenized_query)

        sorted_result = np.argsort(doc_scores)[::-1]
        doc_score = doc_scores[sorted_result].tolist()[:k]
        doc_indices = sorted_result.tolist()[:k]
        return doc_score, doc_indices

    def get_relevant_doc_bulk_bm25(
        self, queries: List, k: Optional[int] = 1
    ) -> Tuple[List, List]:

        tokenized_queries = [self.tokenize_fn(query) for query in queries]
        all_query_terms = set(term for query in tokenized_queries for term in query)

        # frequency matrix for all query terms to all documents
        term_freq_matrix = np.array(
            [
                [doc.get(term, 0) for doc in self.bm25.doc_freqs]
                for term in all_query_terms
            ]
        )

        # calculate IDF vector
        idf_vector = np.array([self.bm25.idf.get(term, 0) for term in all_query_terms])

        doc_scores = []
        doc_indices = []

        for tokenized_query in tqdm(tokenized_queries, desc="Calculating BM25 scores"):
            # 쿼리에서 등장한 단어들에 대한 인덱스를 선택
            query_term_indices = [
                list(all_query_terms).index(term) for term in tokenized_query
            ]

            # 해당 단어들의 문서 내 빈도 추출
            query_term_freqs = term_freq_matrix[query_term_indices, :]

            # 필요한 구성 요소 정의
            idf_terms = idf_vector[query_term_indices][:, np.newaxis]  # IDF 벡터 추출
            k1 = self.bm25.k1
            b = self.bm25.b
            doc_lengths = np.array(self.bm25.doc_len)
            avgdl = self.bm25.avgdl

            # TF(BM25에서의 Term Frequency) 계산
            tf_component = (
                query_term_freqs
                * (k1 + 1)
                / (query_term_freqs + k1 * (1 - b + b * doc_lengths / avgdl))
            )

            # 최종 점수 계산
            doc_scores_for_query = np.sum(idf_terms * tf_component, axis=0)

            # 상위 k개의 문서 선택
            sorted_result = np.argsort(doc_scores_for_query)[::-1]
            doc_scores.append(doc_scores_for_query[sorted_result].tolist()[:k])
            doc_indices.append(sorted_result.tolist()[:k])

        return doc_scores, doc_indices


    def get_relevant_doc_learned(
        self, query: str, k: Optional[int] = 1
    ) -> Tuple[List, List]:
        
        query_vec = self.tokenizer([query], padding=True, truncation=True, return_tensors="pt").to(self.device)
        if query_vec.nnz == 0:
            logger.warning("Query contains only unknown words.")
            raise ValueError(
                "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."
            )
        
        with torch.no_grad(): 
            model_output = self.encoder(**query_vec)
            values = pooling_fn(query_vec, model_output, self.aggregation_method)

            # normalization for cosine similarity
            if self.similarity_metric == "cosine":
                eps = 1e-9
                values = values / (torch.norm(values, dim=-1, keepdim=True) + eps)

            q_embedding = values.squeeze().cpu().numpy().astype(np.float16)

        # 연관 문서 찾기
        similarity_scores = np.dot(q_embedding, self.p_embedding.T)
        if not isinstance(similarity_scores, np.ndarray):
            similarity_scores = similarity_scores.toarray()

        sorted_result = np.argsort(similarity_scores.squeeze())[::-1]
        doc_score = similarity_scores.squeeze()[sorted_result].tolist()[:k]
        doc_indices = sorted_result.tolist()[:k]
        return doc_score, doc_indices


    def get_relevant_doc_bulk_learned(self, queries: List, k: Optional[int] = 1) -> Tuple[List, List]:

        all_embeddings = []
        batch_size = 64
        dataloader = DataLoader(queries, batch_size=batch_size)

        for batch in tqdm(dataloader, desc="Processing Query Batches"):
            query_vecs = self.tokenizer(
                batch, padding=True, truncation=True, return_tensors="pt"
            ).to(self.device)

            with torch.no_grad(): 
                model_output = self.encoder(**query_vecs)
                values = pooling_fn(query_vecs, model_output, self.aggregation_method)

                # normalization for cosine similarity
                if self.similarity_metric == "cosine":
                    eps = 1e-9
                    values = values / (torch.norm(values, dim=-1, keepdim=True) + eps)

                batch_q_embedding = values.squeeze().cpu().numpy() # .astype(np.float16)
                all_embeddings.append(batch_q_embedding)
            # 불필요한 캐시 삭제
            del query_vecs, model_output, values, batch_q_embedding
            torch.cuda.empty_cache()

        q_embedding = np.vstack(all_embeddings).astype(np.float16)

        # 연관 문서 찾기
        similarity_scores = q_embedding.dot(self.p_embedding.T)
        if not isinstance(similarity_scores, np.ndarray):
            similarity_scores = similarity_scores.toarray()

        doc_scores = []
        doc_indices = []

        for i in range(similarity_scores.shape[0]):
            sorted_result = np.argsort(similarity_scores[i, :])[::-1]
            doc_scores.append(similarity_scores[i, :][sorted_result].tolist()[:k])
            doc_indices.append(sorted_result.tolist()[:k])
        return doc_scores, doc_indices


    def evaluate(
        self,
        eval_data_path,
        topk,
        eval_metric: str = "hit",
    ):
        logger.info("Loading evaluation dataset from %s", eval_data_path)
        org_dataset = load_from_disk(eval_data_path)
        eval_df = concatenate_datasets(
            [
                org_dataset["train"].flatten_indices(),
                org_dataset["validation"].flatten_indices(),
            ]
        )
        eval_df = eval_df
        logger.info("Evaluation dataset loaded with %d examples.", len(eval_df))

        result_df = self.retrieve(eval_df, topk=topk)
        if eval_metric == "hit":  # k개의 추천 중 선호 아이템이 있는지 측정
            hit_at_k = hit(result_df)
            logger.info(f"Hit@{topk}: {hit_at_k:.4f}")
            return hit_at_k

        elif eval_metric == "mrr":
            mrr_at_k = mrr(result_df)
            logger.info(f"MRR@{topk}: {mrr_at_k:.4f}")
            return mrr_at_k

        else:
            logger.warning("Unsupported evaluation method: %s", eval_metric)
            return None
