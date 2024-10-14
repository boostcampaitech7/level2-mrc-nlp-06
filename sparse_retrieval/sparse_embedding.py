import json
import os
import pickle
import time
import random
from contextlib import contextmanager
from typing import List, NoReturn, Optional, Tuple, Union

import faiss
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from datasets import Dataset, concatenate_datasets, load_from_disk
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
from tqdm.auto import tqdm

import logging
import sys


logger = logging.getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -    %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")

class SparseRetrieval:
    def __init__(
        self,
        embedding_method,
        tokenize_fn,
        topk,
        use_faiss = False,
        retrieval_data_path: Optional[str] = '../../data/wikipedia_documents.json',
        eval_data_path: Optional[str] = '../../data/train_dataset'
    ) -> NoReturn:

        self.embedding_method = embedding_method
        self.tokenize_fn = tokenize_fn
        self.topk = topk
        self.use_faiss = use_faiss
        self.retrieval_data_path = retrieval_data_path
        self.eval_data_path = eval_data_path
        self.ids = None
        self.contexts = None
        self.indexer = None  # build_faiss()로 생성합니다.

    # wikipedia 데이터 가져오기
    def load_data(self):
        with open(self.retrieval_data_path, "r", encoding='utf-8') as f:
            wiki = json.load(f)
        
        # unique text 추출
        wiki_df = pd.DataFrame(wiki.values())
        wiki_unique_df = wiki_df.drop_duplicates(subset=['text'], keep='first')
        self.ids = wiki_unique_df['document_id'].tolist()
        self.contexts = wiki_unique_df['text'].tolist()
        logger.info(f"Length of unique context: {len(self.contexts)}")


    # TF-IDF 또는 BM25 방식 이용
    def get_sparse_embedding(self) -> NoReturn:

        pickle_name = f"{self.embedding_method}_sparse_embedding.bin"
        vectorizer_name = f"{self.embedding_method}.bin"
        emb_path = os.path.join(pickle_name)
        vectorizer_path = os.path.join(vectorizer_name)

        if self.embedding_method == 'tfidf':
            if os.path.isfile(vectorizer_path) and os.path.isfile(emb_path):
                with open(vectorizer_path, "rb") as file:
                    self.tfidfv = pickle.load(file)
                with open(emb_path, "rb") as file:
                    self.p_embedding = pickle.load(file)
                logger.info("Loaded TF-IDF pickle files.")
            else:
                logger.info("Build TF-IDF passage embedding")
                self.tfidfv = TfidfVectorizer(
                                        tokenizer=self.tokenize_fn.tokenize, 
                                        ngram_range=(1, 2), 
                                        max_features=50000
                                    )
                self.p_embedding = self.tfidfv.fit_transform(tqdm(self.contexts, desc="TF-IDF Vectorization"))

                with open(vectorizer_path, "wb") as file:
                    pickle.dump(self.tfidfv, file)
                with open(emb_path, "wb") as file:
                    pickle.dump(self.p_embedding, file)
                logger.info("Embedding pickle saved.")

        elif self.embedding_method == 'bm25':
            if os.path.isfile(vectorizer_path):  # bm25는 embedding 저장/사용되지 않음
                with open(vectorizer_path, "rb") as file:
                    self.bm25 = pickle.load(file)
                logger.info("Loaded BM25 pickle file.")
            else:
                logger.info("Fit BM25")
                tokenized_corpus = [self.tokenize_fn.tokenize(doc) for doc in tqdm(self.contexts, desc="Tokenizing for BM25")]
                self.bm25 = BM25Okapi(tokenized_corpus)

                with open(vectorizer_path, "wb") as file:
                    pickle.dump(self.bm25, file)
                logger.info("BM25 pickle saved.")
        

    def build_faiss(self, num_clusters=64) -> NoReturn:
        assert self.embedding_method == 'tfidf', 'BM25는 FAISS 적용 불가'
        
        indexer_name = f"{self.embedding_method}_faiss_clusters{num_clusters}.index"
        indexer_path = os.path.join(indexer_name)
        if os.path.isfile(indexer_path):
            logger.info("Load Saved Faiss Indexer.")
            self.indexer = faiss.read_index(indexer_path)
        else:
            logger.info(f"Creating FAISS indexer from embeddings with num_clusters {num_clusters}.")
            p_emb = self.p_embedding.astype(np.float32).toarray()
            emb_dim = p_emb.shape[-1]

            num_clusters = num_clusters
            quantizer = faiss.IndexFlatL2(emb_dim)  ###TODO: L2 외의 다른 Metric 적용
            self.indexer = faiss.IndexIVFScalarQuantizer(
                quantizer, quantizer.d, num_clusters, faiss.METRIC_L2
            )
            with timer("Train FAISS indexer"):
                self.indexer.train(p_emb)
            with timer("Add p_embedddings to FAISS indexer"):
                self.indexer.add(p_emb)
            faiss.write_index(self.indexer, indexer_path)
            print("Faiss Indexer Saved.")

    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1, use_faiss: Optional[bool] = False
    ) -> Union[Tuple[List, List], pd.DataFrame]:

        if self.use_faiss:
            assert self.indexer is not None, "build_faiss() 메소드를 먼저 수행해주세요."
        else:
            if self.embedding_method == 'tfidf':
                assert self.p_embedding is not None, "get_sparse_embedding() 메소드를 먼저 수행해주세요."
            elif self.embedding_method == 'bm25':
                assert self.bm25 is not None, "get_sparse_embedding() 메소드를 먼저 수행해주세요."


        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc(query_or_dataset, k=topk, use_faiss=self.use_faiss)
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print(f"Top-{i+1} passage with score {doc_scores[i]:4f}")
                print(self.contexts[doc_indices[i]])

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):

            # 여러 쿼리에 대한 처리
            queries = query_or_dataset["question"]
            retrieved_data = []  # 결과를 저장할 딕셔너리 리스트

            with timer("query exhaustive search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk(queries, k=topk, use_faiss=self.use_faiss)
            
            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="Sparse retrieval: ")
            ):
                result = {
                    "question": example["question"],
                    "id": example["id"],
                    "context": [self.contexts[pid] for pid in doc_indices[idx]]  # 리스트 형태로 반환
                }
                if "context" in example.keys() and "answers" in example.keys():
                    result["original_context"] = example["context"]
                    result["answers"] = example["answers"]
                retrieved_data.append(result)

            return retrieved_data

    def get_relevant_doc(self, query: str, k: Optional[int] = 1, use_faiss: Optional[bool] = False) -> Tuple[List, List]:

        """
        Arguments:
            query (str):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """

        if self.embedding_method == 'tfidf':
            query_vec = self.tfidfv.transform([query])
            assert (np.sum(query_vec) != 0), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

            if self.use_faiss:
                q_emb = query_vec.toarray().astype(np.float32)
                D, I = self.indexer.search(q_emb, k)
                return D.tolist()[0], I.tolist()[0]
            else:
                result = query_vec * self.p_embedding.T
                if not isinstance(result, np.ndarray):
                    result = result.toarray()

                sorted_result = np.argsort(result.squeeze())[::-1]
                doc_score = result.squeeze()[sorted_result].tolist()[:k]
                doc_indices = sorted_result.tolist()[:k]
                return doc_score, doc_indices
            
        elif self.embedding_method == 'bm25':
            tokenized_query = self.tokenize_fn.tokenize(query)
            doc_scores = self.bm25.get_scores(tokenized_query)

            sorted_result = np.argsort(doc_scores)[::-1]
            doc_score = doc_scores[sorted_result].tolist()[:k]
            doc_indices = sorted_result.tolist()[:k]
            return doc_score, doc_indices


    def get_relevant_doc_bulk(
        self, queries: List, k: Optional[int] = 1, use_faiss: Optional[bool] = False
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

        if self.embedding_method == 'tfidf':
            query_vecs = self.tfidfv.transform(queries)
            assert (np.sum(query_vecs) != 0), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

            if self.use_faiss:
                q_emb = query_vecs.toarray().astype(np.float32)
                D, I = self.indexer.search(q_emb, k)
                return D.tolist(), I.tolist()
            else:
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

            
        elif self.embedding_method == 'bm25':
            doc_scores = []
            doc_indices = []

            for query in tqdm(queries): # bm25는 한 쿼리씩 처리하도록 구성되어 있음. 많이 느림.
                tokenized_query = self.tokenize_fn.tokenize(query)
                doc_scores_for_query = self.bm25.get_scores(tokenized_query)

                # 점수를 기준으로 상위 k개의 문서 선택
                sorted_result = np.argsort(doc_scores_for_query)[::-1]
                doc_scores.append(doc_scores_for_query[sorted_result].tolist()[:k])
                doc_indices.append(sorted_result.tolist()[:k])
            return doc_scores, doc_indices

    def evaluate(self, evaluation_method = 'correct'):
        org_dataset = load_from_disk(self.eval_data_path)
        eval_df = concatenate_datasets(
            [
                org_dataset["train"].flatten_indices(),
                org_dataset["validation"].flatten_indices(),
            ]
        )
        logger.info(eval_df)

        result_list = self.retrieve(eval_df, topk=self.topk, use_faiss=self.use_faiss)
        if evaluation_method == 'correct':
            correct = []
            for example in result_list:
                correct.append(example['original_context'] in example['context'])
            score = sum(correct)/len(correct)
            logger.info(f"Correct retrieval result: {score}")
            return result_list, score

if __name__ == "__main__":

    tokenizer_model = 'klue/bert-base'
    tokenize_fn = AutoTokenizer.from_pretrained(tokenizer_model, use_fast=True)

    use_faiss = False

    retriever = SparseRetrieval(
        embedding_method = "tfidf",  # 'tfidf','bm25'
        tokenize_fn=tokenize_fn,
        topk = 50,
        use_faiss = use_faiss,
        retrieval_data_path = '../../data/wikipedia_documents.json',
        eval_data_path = '../../data/train_dataset'
    )
    retriever.load_data()
    retriever.get_sparse_embedding()
    if use_faiss:
        retriever.build_faiss()
    
    result_list, score = retriever.evaluate(evaluation_method = 'correct')
    print(f"Correct retrieval result: {score}")
