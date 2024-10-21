import sys
import os
import logging
from tqdm.auto import tqdm
from typing import List, NoReturn, Optional, Tuple, Union
import numpy as np
import pandas as pd
import pickle
import torch
from datasets import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
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


# 현재 파일 기준 2단계 상위 경로를 시스템 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils_sparse_retrieval import timer

# logger 지정
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -    %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# TF-IDF, BM25 알고리즘으로 문서 검색
class SparseRetrieval: 

    def __init__(self, embedding_method: str, tokenizer, contexts) -> NoReturn:
        """
        문서 검색을 위한 Sparse Retrieval 클래스. 
        TF-IDF 및 BM25 알고리즘을 사용하여 문서 임베딩과 검색을 수행한다.

        Args:
            embedding_method (str): 임베딩 방법 ('tfidf', 'bm25').
            tokenizer (Callable): 텍스트 토큰화 함수 ('mecab', 허깅페이스 모델 이름)
            contexts (List[str]): 임베딩할 문서 목록
        """
        self.embedding_method = embedding_method
        self.tokenizer = tokenizer
        self.contexts = contexts
        self.tfidfv = None       # TF-IDF 벡터화기
        self.p_embedding = None  # TF-IDF 임베딩 결과
        self.bm25 = None         # BM25 모델


    def get_sparse_embedding_tfidf(self) -> NoReturn:
        """
        TF-IDF 임베딩을 수행하고 결과를 저장한다.
        이미 임베딩 결과가 저장된 파일이 있으면 이를 불러오고, 없으면 새로 생성한다.
        sklearn.feature_extraction.text.TfidfVectorizer() 모듈 사용
        """
        # 임베딩 파일명 및 경로 지정
        sparse_path_name = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        pickle_name = f"{self.embedding_method}_sparse_embedding.bin"
        vectorizer_name = f"{self.embedding_method}.bin"
        emb_path = os.path.join(sparse_path_name, "model", pickle_name)
        vectorizer_path = os.path.join(sparse_path_name, "model", vectorizer_name)

        # 파일이 존재하면 불러오기, 없으면 새로 생성 후 저장
        if os.path.isfile(vectorizer_path) and os.path.isfile(emb_path):
            logger.info("Loading TF-IDF pickle files.")
            with open(vectorizer_path, "rb") as file:
                self.tfidfv = pickle.load(file)
            with open(emb_path, "rb") as file:
                self.p_embedding = pickle.load(file)
        else:
            logger.info("Build TF-IDF passage embedding")
            self.tfidfv = TfidfVectorizer(
                tokenizer=self.tokenizer, ngram_range=(1, 3), max_features=None,
                sublinear_tf=True
            )
            self.p_embedding = self.tfidfv.fit_transform(
                tqdm(self.contexts, desc="TF-IDF Vectorization")
            )
            logger.info("Saving TF-IDF pickle files.")
            
            # 'model' 폴더 없으면 생성
            if not os.path.isdir(os.path.join(sparse_path_name,"model")):
                os.mkdir(os.path.join(sparse_path_name,"model"))

            # vectorizer 및 context 임베딩 결과 저장 (.bin 파일)
            with open(vectorizer_path, "wb") as file:
                pickle.dump(self.tfidfv, file)
            with open(emb_path, "wb") as file:
                pickle.dump(self.p_embedding, file)
            logger.info("Embedding pickle saved.")


    def get_sparse_embedding_bm25(self) -> NoReturn:
        """
        BM25 임베딩을 수행하고 결과를 저장한다.
        이미 BM25 모델이 저장된 파일이 있으면 이를 불러오고, 없으면 새로 생성한다.
        rank_bm25.BM25Okapi() 모듈 사용
        """
        # 모델 파일명 및 경로 지정
        sparse_path_name = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        vectorizer_name = f"{self.embedding_method}.bin"
        vectorizer_path = os.path.join(sparse_path_name, "model", vectorizer_name)

        # BM25 모델 불러오기 또는 새로 학습
        if os.path.isfile(vectorizer_path):
            logger.info("Loading BM25 pickle file.")
            with open(vectorizer_path, "rb") as file:
                self.bm25 = pickle.load(file)
        else:
            logger.info("Fitting BM25 model.")
            self.bm25 = BM25Okapi(tqdm(self.contexts, desc="Tokenizing for BM25"), tokenizer = self.tokenizer, k1=1.0)

            logger.info("Saving BM25 pickle file.")

            # 'model' 폴더 없으면 생성
            if not os.path.isdir(os.path.join(sparse_path_name,"model")):
                os.mkdir(os.path.join(sparse_path_name,"model"))

            # BM25 모델 저장
            with open(vectorizer_path, "wb") as file:
                pickle.dump(self.bm25, file)
            logger.info("BM25 model saved.")

    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 10, save: Optional[bool] = True, retrieval_save_path: Optional[str] = "../outputs/"
    ) -> Union[Tuple[List, List], pd.DataFrame]:
        """
        주어진 쿼리 또는 데이터셋에 대해 문서 검색을 수행한다.

        Args:
            query_or_dataset (Union[str, Dataset]): 검색할 단일 쿼리 또는 데이터셋.
            topk (Optional[int], optional): 반환할 상위 문서 개수. 기본값은 10.
            save (Optional[bool], optional): 검색 결과를 CSV 파일로 저장할지 여부. 기본값은 True.
            retrieval_save_path (Optional[str], optional): 검색 결과 저장 경로. 기본값은 "../outputs/".

        Returns:
            Union[Tuple[List, List], pd.DataFrame]: 검색된 문서와 점수 또는 데이터셋에 대한 검색 결과 데이터프레임.
        """

        # 임베딩 벡터 존재하는지 확인
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

        # 단일 쿼리에 대한 검색
        if isinstance(query_or_dataset, str):
            logger.info("Retrieving for single query.")
            if self.embedding_method == "tfidf":
                doc_scores, doc_indices = self.get_relevant_doc_tfidf(query_or_dataset, k=topk)
            elif self.embedding_method == "bm25":
                doc_scores, doc_indices = self.get_relevant_doc_bm25(query_or_dataset, k=topk)
            
            logger.info(f"[Search query]\n{query_or_dataset}\n")
            for i in range(topk):
                logger.info(f"Top-{i+1} passage with score {doc_scores[i]:.4f}")
                logger.debug(f"Passage: {self.contexts[doc_indices[i]]}")

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        # 데이터셋에 대한 검색
        elif isinstance(query_or_dataset, Dataset):
            logger.info("Retrieving for dataset queries.")
            queries = query_or_dataset["question"]
            retrieved_data = []

            # 유사한 문서 점수(doc_scores), 문서 인덱스(doc_indices) 추출
            with timer("query exhaustive search"):
                if self.embedding_method == "tfidf":
                    doc_scores, doc_indices = self.get_relevant_doc_bulk_tfidf(queries, k=topk)
                elif self.embedding_method == "bm25":
                    doc_scores, doc_indices = self.get_relevant_doc_bulk_bm25(queries, k=topk)

            # 각 쿼리별 검색 결과 딕셔너리로 저장
            for idx, example in enumerate(tqdm(query_or_dataset, desc="Sparse retrieval: ")):
                # 쿼리와 유사한 wiki 문서 리스트
                retrieved_contexts = [self.contexts[pid] for pid in doc_indices[idx]]

                # retrieval 결과 딕셔너리
                retrieved_dict = {
                    "question": example["question"],         # 쿼리
                    "id": example["id"],                     # id
                    "context": " ".join(retrieved_contexts), # 검색된 context 이어 붙이기
                }

                # 정답이 있는 데이터의 경우 원본 context 및 정답 저장 (성능 평가 위함)
                if "context" in example.keys() and "answers" in example.keys():
                    retrieved_dict["original_context"] = example["context"]
                    retrieved_dict["answers"] = example["answers"]
                    # 정답 문서가 몇 번째로 retrieval되었는지 rank 저장 (retrieved 결과에 정답 문서 없으면 0)
                    try:
                        retrieved_dict["rank"] = (retrieved_contexts.index(example["context"]) + 1)
                    except ValueError:
                        retrieved_dict["rank"] = 0 
                retrieved_data.append(retrieved_dict)

            # Pandas DataFrame으로 결과 변환
            retrieved_df = pd.DataFrame(retrieved_data)

            # save=True 이면 지정된 경로에 결과 데이터프레임 csv 파일로 저장
            if save:
                retrieved_file_name = retrieval_save_path + self.embedding_method + '.csv'
                retrieved_df.to_csv(retrieved_file_name, index=False)

            logger.info("Completed retrieval for dataset queries.")

            return retrieved_df


    def get_relevant_doc_tfidf(self, query: str, k: Optional[int] = 10) -> Tuple[List, List]:
        """
        주어진 단일 쿼리에 대해 TF-IDF 방식으로 관련 문서를 검색한다.

        Args:
            query (str): 검색할 쿼리 문장.
            k (Optional[int], optional): 상위 k개의 문서를 반환한다. 기본값은 10이다.

        Raises:
            ValueError: 쿼리에 존재하는 모든 단어가 TF-IDF 벡터라이저의 단어 집합에 없는 경우 오류 발생

        Returns:
            Tuple[List, List]: 문서의 점수와 인덱스를 튜플로 반환한다.
        """

        query_vec = self.tfidfv.transform([query])  # 입력된 쿼리를 TF-IDF 벡터로 변환

        # 쿼리에 존재하는 모든 단어가 TF-IDF 벡터라이저의 단어 집합에 없는 경우 오류 발생
        if query_vec.nnz == 0:
            logger.warning("Query contains only unknown words.")
            raise ValueError(
                "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."
            )

        # 결과를 벡터로 변환하여 문서 간의 유사도를 계산
        result = query_vec * self.p_embedding.T
        if not isinstance(result, np.ndarray):
            result = result.toarray()

        # 유사도 점수를 기준으로 상위 k개의 문서 선택
        sorted_result = np.argsort(result.squeeze())[::-1]
        doc_score = result.squeeze()[sorted_result].tolist()[:k]
        doc_indices = sorted_result.tolist()[:k]
        return doc_score, doc_indices


    def get_relevant_doc_bulk_tfidf(self, queries: List, k: Optional[int] = 10) -> Tuple[List, List]:
        """
        여러 쿼리에 대해 TF-IDF 방식으로 관련 문서를 대량 검색한다.

        Args:
            queries (List): 검색할 여러 쿼리 리스트.
            k (Optional[int], optional): 각 쿼리에 대해 상위 k개의 문서를 반환한다. 기본값은 10이다.

        Raises:
            ValueError: 쿼리에 존재하는 모든 단어가 TF-IDF 벡터라이저의 단어 집합에 없는 경우 오류 발생.

        Returns:
            Tuple[List, List]: 각 쿼리에 대한 문서의 점수와 인덱스를 리스트 형태로 반환한다.
        """
        # 쿼리 목록을 TF-IDF 벡터로 변환
        query_vecs = self.tfidfv.transform(queries)

        # 쿼리에 존재하는 모든 단어가 TF-IDF 벡터라이저의 단어 집합에 없는 경우 오류 발생
        if query_vecs.nnz == 0:
            logger.warning("One or more queries contain only unknown words.")
            raise ValueError(
                "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."
            )

        # 각 쿼리에 대해 문서 유사도 계산
        result = query_vecs * self.p_embedding.T
        if not isinstance(result, np.ndarray):
            result = result.toarray()

        # 각 쿼리별로 유사한 k개 문서의 유사도 점수, 문서 인덱스 반환
        doc_scores = []
        doc_indices = []
        for i in range(result.shape[0]):
            sorted_result = np.argsort(result[i, :])[::-1]
            doc_scores.append(result[i, :][sorted_result].tolist()[:k])
            doc_indices.append(sorted_result.tolist()[:k])
        return doc_scores, doc_indices


    def get_relevant_doc_bm25(self, query: str, k: Optional[int] = 10) -> Tuple[List, List]:
        """
        주어진 쿼리에 대해 BM25 방식으로 관련 문서를 검색한다.

        Args:
            query (str): 검색할 쿼리 문장.
            k (Optional[int], optional): 상위 k개의 문서를 반환한다. 기본값은 10이다.

        Returns:
            Tuple[List, List]: 문서의 점수와 인덱스를 튜플로 반환한다.
        """

        tokenized_query = self.tokenizer(query)             # 쿼리를 토큰화
        doc_scores = self.bm25.get_scores(tokenized_query)  # BM25 점수 계산

        # 상위 k개의 문서 선택
        sorted_result = np.argsort(doc_scores)[::-1]
        doc_score = doc_scores[sorted_result].tolist()[:k]
        doc_indices = sorted_result.tolist()[:k]
        return doc_score, doc_indices

    def get_relevant_doc_bulk_bm25(self, queries: List, k: Optional[int] = 10) -> Tuple[List, List]:
        """
        여러 쿼리에 대해 BM25 방식으로 관련 문서를 대량 검색한다.

        Args:
            queries (List): 검색할 여러 쿼리 목록.
            k (Optional[int], optional): 각 쿼리에 대해 상위 k개의 문서를 반환한다. 기본값은 10이다.

        Returns:
            Tuple[List, List]: 각 쿼리에 대한 문서의 점수와 인덱스를 리스트 형태로 반환한다.
        """

        tokenized_queries = [self.tokenizer(query) for query in queries]  # 각 쿼리를 토큰화
        all_query_terms = set(term for query in tokenized_queries for term in query)

        # 모든 문서에 대해 쿼리 term의 Frequency Matrix 생성
        term_freq_matrix = np.array(
            [
                [doc.get(term, 0) for doc in self.bm25.doc_freqs]
                for term in all_query_terms
            ]
        )

        # IDF 벡터 계산
        idf_vector = np.array([self.bm25.idf.get(term, 0) for term in all_query_terms])

        doc_scores = []
        doc_indices = []

        # 각 쿼리에 대해 BM25 점수 계산
        for tokenized_query in tqdm(tokenized_queries, desc="Calculating BM25 scores"):
            # 쿼리에서 등장한 단어들에 대한 인덱스를 선택
            query_term_indices = [list(all_query_terms).index(term) for term in tokenized_query]

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



class LearnedSparseRetrieval:
    def __init__(self, embedding_method, embedding_model_name, contexts, ids, collection_name, bgem3_type="sparse", dense_metric_type="IP") -> NoReturn:
        """
        문서 검색을 위한 Learned Sparse Retrieval 클래스. 
        SPLADE 및 BGE-M3 알고리즘을 사용하여 문서 임베딩과 검색을 수행한다.

        Args:
            embedding_method (str): 임베딩 방법 ("splade", "bge-m3").
            embedding_model_name (str): 사용할 임베딩 모델의 이름. (허깅페이스 모델)
            contexts (list): 임베딩할 문서 리스트
            ids (list): 임베딩할 문서의 document_id 리스트
            collection_name (str): Milvus Database에서 사용할 컬렉션의 이름.
            bgem3_type (str, optional): BGE-M3의 유형 ("sparse", "dense", "hybrid").
            dense_metric_type (str, optional): Dense 임베딩에 사용할 메트릭 유형. ("L2", "COSINE", "IP")
        """
        self.embedding_method = embedding_method
        self.embedding_model_name = embedding_model_name
        self.contexts = contexts
        self.ids = ids 
        self.collection_name = collection_name
        self.bgem3_type = bgem3_type
        self.dense_metric_type = dense_metric_type
        self.ef = None   # 임베딩 함수 초기화
        self.col = None  # 컬렉션 초기화

    def get_sparse_embedding_splade(self) -> NoReturn:
        """
        SPLADE 모델을 사용하여 Sparse 임베딩을 생성하고 데이터베이스에 저장한다.
        이미 임베딩 결과가 저장된 컬렉션이 있으면 이를 불러오고, 없으면 새로 생성한다.
        """
        # 저장 경로 지정: Milvus에 연결 ("wiki_embedding.db")
        sparse_path_name = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        connections.connect(uri=f"{sparse_path_name}/model/wiki_embedding.db")

        # 모델 지정
        self.ef = SpladeEmbeddingFunction(
            model_name=self.embedding_model_name, 
            device="cuda:0"  # GPU 사용
        )

        # 컬렉션 조회: 존재하지 않으면 생성
        col_name = self.collection_name
        if utility.has_collection(col_name):
            self.col = Collection(col_name)
            index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
            self.col.create_index("sparse_vector", index)  # 인덱스 생성
            self.col.load()  # 컬렉션 로드
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
            self.col = Collection(col_name, schema, consistency_level="Strong")  # 컬렉션 생성
            index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
            self.col.create_index("sparse_vector", index)  # 인덱스 생성
            self.col.load()  # 컬렉션 로드

            # 임베딩 생성 
            batch_size = 4  # 배치 크기 설정
            for i in tqdm(range(0, len(self.contexts), batch_size)):
                end_idx = min(i + batch_size, len(self.contexts))  # 마지막 배치 처리

                batch_ids = self.ids[i:end_idx]
                batch_contexts = self.contexts[i:end_idx]

                torch.cuda.empty_cache()
                with torch.no_grad():
                    batch_docs_embeddings = self.ef.encode_documents(batch_contexts)  # 문서 임베딩 생성
                    batch_entities = []
                    for j in range(len(batch_ids)):
                        entity = {
                            "id": batch_ids[j],
                            "context": batch_contexts[j],
                            "sparse_vector": batch_docs_embeddings[[j], :],
                        }
                        batch_entities.append(entity)

                    self.col.insert(batch_entities)  # 엔티티를 컬렉션에 삽입
            logger.info(f"Number of entities inserted: {self.col.num_entities}")

    def get_sparse_embedding_bgem3(self) -> NoReturn:
        """
        BGEM3 모델을 사용하여 Sparse 및 Dense 임베딩을 생성하고 데이터베이스에 저장한다.
        이미 임베딩 결과가 저장된 컬렉션이 있으면 이를 불러오고, 없으면 새로 생성한다.
        """
        # 저장 경로 지정: Milvus에 연결 ("wiki_embedding.db")
        sparse_path_name = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        connections.connect(uri=f"{sparse_path_name}/model/wiki_embedding.db")

        # 모델 지정
        self.ef = BGEM3EmbeddingFunction(
            model_name=self.embedding_model_name, # 모델 이름 지정
            device="cuda:0",    # GPU 사용
            use_fp16=True,
            return_sparse=True, # Sparse Embedding 결과 얻기
            return_dense=True   # Dense Embedding 결과 얻기
        )
        dense_dim = self.ef.dim['dense']  # Dense Embedding 차원

        # 컬렉션 조회: 존재하지 않으면 생성
        col_name = self.collection_name
        if utility.has_collection(col_name):
            self.col = Collection(col_name)
            index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
            self.col.create_index("sparse_vector", index)  # sparse 벡터 인덱스 생성
            index = {"index_type": "FLAT", "metric_type": self.dense_metric_type} 
            self.col.create_index("dense_vector", index)   # dense 벡터 인덱스 생성
            self.col.load()  # 컬렉션 로드
            logger.info(f"Collection \"{self.collection_name}\" is loaded.")
            logger.info(f"Number of entities contained: {self.col.num_entities}")
        else:
            # 데이터 스키마 지정
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True), 
                FieldSchema(name="context", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
                FieldSchema(name="dense_vector", dtype=DataType.FLOAT16_VECTOR, dim=dense_dim)
            ]
            schema = CollectionSchema(fields, description="Collection for BGE M3 embeddings")
            self.col = Collection(col_name, schema, consistency_level="Strong")
            index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
            self.col.create_index("sparse_vector", index)  # sparse 벡터 인덱스 생성
            index = {"index_type": "FLAT", "metric_type": self.dense_metric_type} 
            self.col.create_index("dense_vector", index)   # dense 벡터 인덱스 생성
            self.col.load()  # 컬렉션 로드

            # 임베딩 생성 
            batch_size = 4  # 배치 크기 설정
            for i in tqdm(range(0, len(self.contexts), batch_size)):
                end_idx = min(i + batch_size, len(self.contexts))  # 마지막 배치 처리

                batch_ids = self.ids[i:end_idx]
                batch_contexts = self.contexts[i:end_idx]

                torch.cuda.empty_cache()
                with torch.no_grad():
                    batch_docs_embeddings = self.ef.encode_documents(batch_contexts)  # 문서 임베딩 생성
                    batch_sparse_embeddings = batch_docs_embeddings['sparse']  # sparse embedding
                    batch_dense_embeddings = batch_docs_embeddings['dense']    # dense embedding

                    batch_entities = []
                    for j in range(len(batch_ids)):
                        entity = {
                            "id": batch_ids[j],
                            "context": batch_contexts[j],
                            "sparse_vector": batch_sparse_embeddings[[j], :],  # sparse vector
                            "dense_vector": batch_dense_embeddings[j]          # dense vector
                        }
                        batch_entities.append(entity)

                    self.col.insert(batch_entities)  # collection에 entity 삽입
            logger.info(f"Number of entities inserted:{self.col.num_entities}")

    def retrieve(
        self, query_or_dataset: Union[str, Dataset], dense_metric_type = "IP", topk = 10, save: Optional[bool] = True, retrieval_save_path: Optional[str] = "../outputs/"
    ) -> Union[Tuple[List, List], pd.DataFrame]:
        """
        주어진 쿼리 또는 데이터셋에 대해 문서 검색을 수행한다.

        Args:
            query_or_dataset (Union[str, Dataset]): 검색할 단일 쿼리 또는 데이터셋.
            dense_metric_type (str, optional): Dense 임베딩 검색에 사용할 메트릭 유형 ("L2", "COSINE", "IP")
            topk (int, optional): 반환할 상위 k개의 결과 수. 기본값은 10.
            save (Optional[bool], optional):결과를 CSV 파일로 저장할지 여부. 기본값은 True.
            retrieval_save_path (Optional[str], optional): 검색 결과 저장 경로. 기본값은 "../outputs/".

        Returns:
            Union[Tuple[List, List], pd.DataFrame]: 검색된 문서와 점수 또는 데이터셋에 대한 검색 결과 데이터프레임.
        """
        # 임베딩 함수 및 컬렉션이 지정되었는지 확인
        assert self.ef is not None and self.col is not None, "get_sparse_embedding() 메소드를 먼저 수행해주세요."

        # 단일 쿼리에 대한 검색
        if isinstance(query_or_dataset, str):
            query = [query_or_dataset]                       # 쿼리 리스트로 반환
            query_embeddings = self.ef.encode_queries(query) # 쿼리 임베딩 생성
            if self.embedding_method=="splade":
                search_result = self.sparse_search(query_embeddings, topk)  # SPLADE 모델에 대해서는 sparse retrieval
                doc_scores = []
                doc_contexts = []
                for i in range(len(search_result)):
                    doc_scores.append(search_result[i].score)  # 문서 유사도 점수 저장
                    doc_contexts.append(search_result[i].get('context'))  # 문서 context로 저장

                logger.info(f"[Search query]\n{query_or_dataset}\n")
                for i in range(topk):
                    logger.info(f"Top-{i+1} passage with score {doc_scores[i]:.4f}")
                    logger.debug(f"Passage: {doc_contexts[i]}")

                return (doc_scores, doc_contexts)
            
            elif self.embedding_method=="bge-m3":
                if self.bgem3_type == "sparse":
                    # BGE-M3의 sparse retrieval
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
                    # BGE-M3의 dense retrieval
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
                    # BGE-M3의 hybrid retrieval
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

        # 데이터셋에 대한 검색
        elif isinstance(query_or_dataset, Dataset):
            logger.info("Retrieving for dataset queries.")
            queries = query_or_dataset["question"]  # 쿼리 리스트 추출
            query_embeddings = self.ef.encode_queries(queries)  # 모든 쿼리의 임베딩 생성
            if self.embedding_method == "splade":
                search_result = self.sparse_search(query_embeddings, topk)  # SPLADE sparse retrieval
            elif self.embedding_method == "bge-m3":
                if self.bgem3_type == "sparse":
                    search_result = self.sparse_search(query_embeddings['sparse'], topk)  # BGE-M3 sparse retrieval
                elif self.bgem3_type == "dense":
                    search_result = self.dense_search(query_embeddings['dense'], dense_metric_type, topk) # BGE-M3 dense retrieval
                elif self.bgem3_type == "hybrid":
                    search_result = self.hybrid_search(  # BGE-M3 hybrid retrieval
                        query_embeddings['dense'], query_embeddings['sparse'], dense_metric_type, 
                        sparse_weight=1.0, dense_weight=1.0, topk=topk
                    )
            
            retrieved_data = []

            # 각 쿼리별 검색 결과 딕셔너리로 저장
            for idx, example in enumerate(query_or_dataset):
                # 쿼리와 유사한 wiki 문서 리스트
                retrieved_contexts = [hit.get("context") for hit in search_result[idx]]

                # retrieval 결과 딕셔너리
                retrieved_dict = {
                    "question": example['question'],         # 쿼리
                    "id": example['id'],                     # id
                    "context": " ".join(retrieved_contexts)  # 검색된 context 이어 붙이기
                }

                # 정답이 있는 데이터의 경우 원본 context 및 정답 저장 (성능 평가 위함)
                if "context" in example.keys() and "answers" in example.keys():
                    retrieved_dict["original_context"] = example["context"]
                    retrieved_dict["answers"] = example["answers"]
                    # 정답 문서가 몇 번째로 retrieval되었는지 rank 저장 (retrieved 결과에 정답 문서 없으면 0)
                    try:
                        retrieved_dict["rank"] = (
                            retrieved_contexts.index(example["context"]) + 1
                        )
                    except ValueError:
                        retrieved_dict["rank"] = 0  # 정답 문서가 없으면 0

                retrieved_data.append(retrieved_dict)

            # Pandas DataFrame으로 결과 변환
            retrieved_df = pd.DataFrame(retrieved_data)

            # save=True 이면 지정된 경로에 결과 데이터프레임 csv 파일로 저장
            if save:
                retrieved_file_name = retrieval_save_path + self.embedding_method + '_retrieved_df.csv'
                retrieved_df.to_csv(retrieved_file_name, index=False)

            logger.info("Completed retrieval for dataset queries.")

            return retrieved_df
        

    def dense_search(self, query_dense_embedding, dense_metric_type="IP", topk=10):
        """
        Dense Embedding을 사용해 검색한다.

        Args:
            query_dense_embedding: 검색할 dense embedding
            dense_metric_type (str, optional): dense embedding에 사용할 메트릭 유형 ("L2", "COSINE", "IP")
            topk (int, optional): 반환할 상위 k개의 결과 수. 기본값은 10.

        Returns:
            res: 검색 결과
        """
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
        """
        Sparse Embedding을 사용해 검색한다.

        Args:
            query_sparse_embedding: 검색할 sparse embedding
            topk (int, optional): 반환할 상위 k개의 결과 수. 기본값은 10.

        Returns:
            res: 검색 결과
        """
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
        """
        Hybrid Embedding을 사용해 검색한다.
        Sparse 및 Dense 임베딩을 조합해 결과를 얻는다.

        Args:
            query_dense_embedding: 검색할 dense embedding
            query_sparse_embedding: 검색할 sparse embedding
            dense_metric_type (str, optional): dense embedding에 사용할 메트릭 유형 ("L2", "COSINE", "IP")
            sparse_weight (float, optional): sparse retrieval의 가중치. 기본값은 1.0
            dense_weight (float, optional): dense retrieval의 가중치. 기본값은 1.0
            topk (int, optional): 반환할 상위 k개의 결과 수. 기본값은 10.

        Returns:
            res: 검색 결과
        """

        # dense retrieval을 위한 parameter 설정
        dense_search_params = {"metric_type": dense_metric_type, "params": {}}
        dense_req = AnnSearchRequest(
            query_dense_embedding, "dense_vector", dense_search_params, limit=topk
        )

        # sparse retrieval을 위한 parameter 설정
        sparse_search_params = {"metric_type": "IP", "params": {}}
        sparse_req = AnnSearchRequest(
            query_sparse_embedding, "sparse_vector", sparse_search_params, limit=topk
        )

        # 가중치를 사용하여 재순위할 reranker 생성
        rerank = WeightedRanker(sparse_weight, dense_weight)

        # 하이브리드 검색 수행
        res = self.col.hybrid_search(
            [sparse_req, dense_req], rerank=rerank, limit=topk, output_fields=["context"]
        )
        
        return res
