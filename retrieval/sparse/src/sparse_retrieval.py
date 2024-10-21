import logging
import sys
import os
import pickle
from typing import List, NoReturn, Optional, Tuple, Union

import numpy as np
import pandas as pd
from datasets import Dataset, concatenate_datasets, load_from_disk
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
from tqdm.auto import tqdm

# 2단계 상위 경로를 시스템 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils_sparse_retrieval import timer, hit, mrr

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -    %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class SparseRetrieval:
    def __init__(self, embedding_method: str, tokenizer, contexts, document_ids) -> NoReturn:

        self.embedding_method = embedding_method
        self.tokenize_fn = tokenizer.tokenize
        self.contexts = contexts
        self.document_ids = document_ids
        self.tfidfv = None
        self.p_embedding = None
        self.bm25 = None

    def get_sparse_embedding_tfidf(self) -> NoReturn:
        sparse_path_name = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        pickle_name = f"{self.embedding_method}_sparse_embedding.bin"
        vectorizer_name = f"{self.embedding_method}.bin"
        emb_path = os.path.join(sparse_path_name, "model", pickle_name)
        vectorizer_path = os.path.join(sparse_path_name, "model", vectorizer_name)

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
        sparse_path_name = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        vectorizer_name = f"{self.embedding_method}.bin"
        vectorizer_path = os.path.join(sparse_path_name, "model", vectorizer_name)

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

    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1, save: Optional[bool] = True, retrieval_save_path: Optional[str] = ""
    ) -> Union[Tuple[List, List], pd.DataFrame]:

        if self.embedding_method == "tfidf":
            assert (
                self.p_embedding is not None
            ), "get_sparse_embedding() 메소드를 먼저 수행해주세요."
            logger.debug("Using TF-IDF for retrieval.")
        elif self.embedding_method == "bm25":
            assert (
                self.bm25 is not None
            ), "get_sparse_embedding() 메소드를 먼저 수행해주세요."
            logger.debug("Using BM25 for retrieval.")

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

            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="Sparse retrieval: ")
            ):
                retrieved_contexts = [self.contexts[pid] for pid in doc_indices[idx]]
                retrieved_document_ids = [self.document_ids[pid] for pid in doc_indices[idx]]

                retrieved_dict = {
                    # 쿼리와 해당 ID 반환
                    "question": example["question"],
                    "id": example["id"],
                    "document_id": retrieved_document_ids,
                    # retrieve한 passage의 context 이어 붙이기
                    # "context": " ".join(retrieved_contexts),
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
            if save:
                retrieved_df.to_csv(retrieval_save_path, index=False)
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
