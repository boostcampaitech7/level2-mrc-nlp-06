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
import argparse
import csv


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -    %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

@contextmanager
def timer(name):
    t0 = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - t0
        logger.info(f"[{name}] done in {elapsed:.3f} s")


# file size
def sizeof_fmt(num, suffix="B"):
    for unit in ["", "K", "M", "G", "T", "P", "E", "Z"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f} {unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f} Y{suffix}"


class SparseRetrieval:
    def __init__(
        self,
        embedding_method: str,
        tokenizer,
        topk: int,
        use_faiss: bool = False,
        retrieval_data_path: Optional[str] = '../../data/wikipedia_documents.json',
        eval_data_path: Optional[str] = '../../data/train_dataset'
    ) -> NoReturn:

        self.embedding_method = embedding_method
        self.tokenize_fn = tokenizer.tokenize
        self.topk = topk
        self.use_faiss = use_faiss
        self.retrieval_data_path = retrieval_data_path
        self.eval_data_path = eval_data_path
        self.contexts = None
        self.indexer = None  # build_faiss()로 생성합니다.
        self.tfidfv = None
        self.p_embedding = None
        self.bm25 = None

    # load 'wikipedia_documents.json' data
    def load_data(self):
        logger.info("Loading Wikipedia data from %s", self.retrieval_data_path)
        if os.path.isfile(self.retrieval_data_path):
            file_size = sizeof_fmt(os.path.getsize(self.retrieval_data_path))
            logger.info("File size of %s: %s", self.retrieval_data_path, file_size)
        else:
            logger.warning("File %s does not exist.", self.retrieval_data_path)

        with open(self.retrieval_data_path, "r", encoding='utf-8') as f:
            wiki = json.load(f)
        
        # unique text 추출
        wiki_df = pd.DataFrame(wiki.values())
        wiki_unique_df = wiki_df.drop_duplicates(subset=['text'], keep='first')
        self.contexts = wiki_unique_df['text'].tolist()
        logger.info(f"Length of unique context: {len(self.contexts)}")


    # Generate sparse embedding for contexts (methods: TF-IDF | BM25)
    def get_sparse_embedding(self) -> NoReturn:
        pickle_name = f"{self.embedding_method}_sparse_embedding.bin"
        vectorizer_name = f"{self.embedding_method}.bin"
        emb_path = os.path.join(pickle_name)
        vectorizer_path = os.path.join(vectorizer_name)

        if self.embedding_method == 'tfidf':
            if os.path.isfile(vectorizer_path) and os.path.isfile(emb_path):
                logger.info("Loading TF-IDF pickle files.")
                logger.info("File size of %s: %s", vectorizer_path, sizeof_fmt(os.path.getsize(vectorizer_path)))
                logger.info("File size of %s: %s", emb_path, sizeof_fmt(os.path.getsize(emb_path)))
                with open(vectorizer_path, "rb") as file:
                    self.tfidfv = pickle.load(file)
                with open(emb_path, "rb") as file:
                    self.p_embedding = pickle.load(file)
            else:
                logger.info("Build TF-IDF passage embedding")
                self.tfidfv = TfidfVectorizer(
                    tokenizer=self.tokenize_fn, 
                    ngram_range=(1, 2), 
                    max_features=50000
                )
                with timer("TF-IDF Vectorization"):
                    self.p_embedding = self.tfidfv.fit_transform(
                        tqdm(self.contexts, desc="TF-IDF Vectorization")
                    )

                logger.info("Saving TF-IDF pickle files.")
                with open(vectorizer_path, "wb") as file:
                    pickle.dump(self.tfidfv, file)
                logger.info("File %s saved with size %s", vectorizer_path, sizeof_fmt(os.path.getsize(vectorizer_path)))
                with open(emb_path, "wb") as file:
                    pickle.dump(self.p_embedding, file)
                logger.info("File %s saved with size %s", emb_path, sizeof_fmt(os.path.getsize(emb_path)))
                logger.info("Embedding pickle saved.")

        elif self.embedding_method == 'bm25':
            if os.path.isfile(vectorizer_path):  # bm25 does not use p_embedding
                logger.info("Loading BM25 pickle file.")
                logger.info("File size of %s: %s", vectorizer_path, sizeof_fmt(os.path.getsize(vectorizer_path)))
                with open(vectorizer_path, "rb") as file:
                    self.bm25 = pickle.load(file)
            else:
                logger.info("Fitting BM25 model.")
                tokenized_corpus = [self.tokenize_fn(doc) for doc in tqdm(self.contexts, desc="Tokenizing for BM25")]
                self.bm25 = BM25Okapi(tokenized_corpus)

                logger.info("Saving BM25 pickle file.")
                with open(vectorizer_path, "wb") as file:
                    pickle.dump(self.bm25, file)
                logger.info("File %s saved with size %s", vectorizer_path, sizeof_fmt(os.path.getsize(vectorizer_path)))
                logger.info("BM25 model saved.")
        

    def build_faiss(self, num_clusters=64) -> NoReturn:
        assert self.embedding_method == 'tfidf', 'BM25는 FAISS 적용 불가'
        
        indexer_name = f"{self.embedding_method}_faiss_clusters{num_clusters}.index"
        indexer_path = os.path.join(indexer_name)
        if os.path.isfile(indexer_path):
            logger.info("Loading saved Faiss indexer from %s.", indexer_path)
            logger.info("File size of %s: %s", indexer_path, sizeof_fmt(os.path.getsize(indexer_path)))
            self.indexer = faiss.read_index(indexer_path)
        else:
            logger.info(f"Creating FAISS indexer with {num_clusters} clusters.")
            p_emb = self.p_embedding.astype(np.float32).toarray()
            emb_dim = p_emb.shape[-1]

            quantizer = faiss.IndexFlatL2(emb_dim)
            self.indexer = faiss.IndexIVFScalarQuantizer(
                quantizer, quantizer.d, num_clusters, faiss.METRIC_L2
            )
            with timer("Train FAISS indexer"):
                self.indexer.train(p_emb)
            with timer("Add p_embedddings to FAISS indexer"):
                self.indexer.add(p_emb)
            faiss.write_index(self.indexer, indexer_path)
            logger.info("Faiss indexer saved to %s.", indexer_path)
            logger.info("File size of %s: %s", indexer_path, sizeof_fmt(os.path.getsize(indexer_path)))

    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:

        if self.use_faiss:
            assert self.indexer is not None, "build_faiss() 메소드를 먼저 수행해주세요."
            logger.debug("Using FAISS for retrieval.")
        else:
            if self.embedding_method == 'tfidf':
                assert self.p_embedding is not None, "get_sparse_embedding() 메소드를 먼저 수행해주세요."
                logger.debug("Using TF-IDF for retrieval.")
            elif self.embedding_method == 'bm25':
                assert self.bm25 is not None, "get_sparse_embedding() 메소드를 먼저 수행해주세요."
                logger.debug("Using BM25 for retrieval.")


        if isinstance(query_or_dataset, str):
            logger.info("Retrieving for single query.")
            doc_scores, doc_indices = self.get_relevant_doc(query_or_dataset, k=topk)
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
                doc_scores, doc_indices = self.get_relevant_doc_bulk(queries, k=topk)
            
            for idx, example in enumerate(tqdm(query_or_dataset, desc="Sparse retrieval: ")):
                result = {
                    "question": example["question"],
                    "id": example["id"],
                    "context": [self.contexts[pid] for pid in doc_indices[idx]]  # format: list
                }
                if "context" in example.keys() and "answers" in example.keys():
                    result["original_context"] = example["context"]
                    result["answers"] = example["answers"]
                retrieved_data.append(result)

            logger.info("Completed retrieval for dataset queries.")
            return retrieved_data

    def get_relevant_doc(self, query: str, k: Optional[int] = 1) -> Tuple[List, List]:

        if self.embedding_method == 'tfidf':
            query_vec = self.tfidfv.transform([query])
            if query_vec.nnz == 0:
                logger.warning("Query contains only unknown words.")
                raise ValueError("오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다.")

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
            tokenized_query = self.tokenize_fn(query)
            doc_scores = self.bm25.get_scores(tokenized_query)

            sorted_result = np.argsort(doc_scores)[::-1]
            doc_score = doc_scores[sorted_result].tolist()[:k]
            doc_indices = sorted_result.tolist()[:k]
            return doc_score, doc_indices


    def get_relevant_doc_bulk(self, queries: List, k: Optional[int] = 1) -> Tuple[List, List]:

        if self.embedding_method == 'tfidf':
            query_vecs = self.tfidfv.transform(queries)
            if query_vecs.nnz == 0:
                logger.warning("One or more queries contain only unknown words.")
                raise ValueError("오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다.")

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
            tokenized_queries = [self.tokenize_fn(query) for query in queries]
            all_query_terms = set(term for query in tokenized_queries for term in query)

            # frequency matrix for all query terms to all documents
            term_freq_matrix = np.array([[doc.get(term, 0) for doc in self.bm25.doc_freqs] for term in all_query_terms])

            # calculate IDF vector
            idf_vector = np.array([self.bm25.idf.get(term, 0) for term in all_query_terms])

            doc_scores = []
            doc_indices = []

            for tokenized_query in tqdm(tokenized_queries, desc="Calculating BM25 scores"):
                # 쿼리에서 등장한 단어들에 대한 인덱스를 선택
                query_term_indices = [list(all_query_terms).index(term) for term in tokenized_query]

                # 해당 단어들의 문서 내 빈도 추출
                query_term_freqs = term_freq_matrix[query_term_indices, :]

                # 벡터화된 BM25 점수 계산
                doc_scores_for_query = np.sum(
                    (idf_vector[query_term_indices][:, np.newaxis] *
                      (query_term_freqs * (self.bm25.k1 + 1) /
                        (query_term_freqs + self.bm25.k1 * (1 - self.bm25.b + self.bm25.b * np.array(self.bm25.doc_len) / self.bm25.avgdl)))),
                    axis=0
                )

                # 상위 k개의 문서 선택
                sorted_result = np.argsort(doc_scores_for_query)[::-1]
                doc_scores.append(doc_scores_for_query[sorted_result].tolist()[:k])
                doc_indices.append(sorted_result.tolist()[:k])

            return doc_scores, doc_indices
        

    def evaluate(self, evaluation_method: str = 'hit', output_csv: Optional[str] = None, inference_time: Optional[float] = None, total_time: Optional[float] = None):
        logger.info("Loading evaluation dataset from %s", self.eval_data_path)
        if os.path.isdir(self.eval_data_path):
            org_dataset = load_from_disk(self.eval_data_path)
            eval_df = concatenate_datasets(
                [
                    org_dataset["train"].flatten_indices(),
                    org_dataset["validation"].flatten_indices(),
                ]
            )
            logger.info("Evaluation dataset loaded with %d examples.", len(eval_df))
        else:
            logger.error("Evaluation data path %s does not exist or is not a directory.", self.eval_data_path)
            raise FileNotFoundError(f"Evaluation data path {self.eval_data_path} not found.")

        result_list = self.retrieve(eval_df, topk=self.topk)
        if evaluation_method == 'hit':  # k개의 추천 중 선호 아이템이 있는지 측정
            logger.info("Evaluating Hit@k.")
            hits = 0
            for example in result_list:
                if 'original_context' in example and example['original_context'] in example.get('context', []):
                    hits += 1
            hit_at_k = hits / len(result_list) if len(result_list) > 0 else 0.0
            logger.info(f"Hit@{self.topk}: {hit_at_k:.4f}")
            return hit_at_k
        elif evaluation_method == 'mrr':
            logger.info("Evaluating MRR@k.")
            mrr_total = 0.0
            for example in result_list:
                if 'original_context' in example:
                    try:
                        rank = example['context'].index(example['original_context']) + 1
                        if rank <= self.topk:
                            mrr_total += 1.0 / rank
                    except ValueError:
                        pass  # original_context not in retrieved contexts
            mrr_at_k = mrr_total / len(result_list) if len(result_list) > 0 else 0.0
            logger.info(f"MRR@{self.topk}: {mrr_at_k:.4f}")
            return mrr_at_k
        else:
            logger.warning("Unsupported evaluation method: %s", evaluation_method)
            return None


def parse_arguments():
    parser = argparse.ArgumentParser(description="Sparse Retrieval Evaluation Script")
    parser.add_argument('--embedding_method', type=str, choices=['tfidf', 'bm25'], required=True, help="Embedding method to use: 'tfidf' or 'bm25'")
    parser.add_argument('--topk', type=int, required=True, help="Number of top documents to retrieve")
    parser.add_argument('--evaluation_methods', type=str, nargs='+', default=['hit', 'mrr'], help="Evaluation methods to use: 'hit', 'mrr'")
    return parser.parse_args()

def append_to_csv(output_csv: str, row: dict, headers: List[str]):
    file_exists = os.path.isfile(output_csv)
    with open(output_csv, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


if __name__ == "__main__":
    args = parse_arguments()

    # Initialize tokenizer
    logger.info("Starting Sparse Retrieval System.")
    tokenizer_model = 'klue/bert-base'
    logger.info("Loading tokenizer model: %s", tokenizer_model)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, use_fast=True)

    logger.info("Embedding method: %s", args.embedding_method)
    logger.info("Top-K: %d", args.topk)
    logger.info("Evaluation method: %s", args.evaluation_methods)

    retriever = SparseRetrieval(
        embedding_method = args.embedding_method,  # 'tfidf','bm25'
        tokenizer=tokenizer,
        topk = args.topk,
        use_faiss = False,
        retrieval_data_path = '../../data/wikipedia_documents.json',
        eval_data_path = '../../data/train_dataset'
    )

    # Record total start time
    total_start_time = time.time()

    # Load data
    logger.info("Loading data.")
    with timer("Load data"):
        retriever.load_data()

    # Generate embeddings
    logger.info("Generating sparse embeddings.")
    with timer("Generate sparse embeddings"):
        retriever.get_sparse_embedding()

    # Build FAISS index if needed
    # if use_faiss:
    #     logger.info("Building FAISS index.")
    #     with timer("Build FAISS index"):
    #         retriever.build_faiss()
    
    # Evaluate
    logger.info("Evaluating retrieval performance.")
    evaluation_results = {}
    inference_times = {}
    for eval_method in args.evaluation_methods:
        with timer(f"Evaluation {eval_method}"):
            result = retriever.evaluate(evaluation_method=eval_method)
            evaluation_results[eval_method] = result

    # Record total end time
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    logger.info(f"Total execution time: {total_time:.3f} s")

    # Prepare row for CSV
    row = {
        "embedding_method": args.embedding_method,
        "topk": args.topk,
        "total_time_sec": f"{total_time:.3f}"
    }

    for eval_method, score in evaluation_results.items():
        row[f"{eval_method}@k"] = f"{score:.4f}"

    # Append to CSV
    headers = ["embedding_method", "topk", "total_time_sec"] + [f"{method}@k" for method in args.evaluation_methods]
    append_to_csv("test_sparse_embedding.csv", row, headers)

    logger.info("Sparse Retrieval System finished execution.")
