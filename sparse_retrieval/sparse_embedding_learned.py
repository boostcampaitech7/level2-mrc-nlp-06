import logging
import json
import sys
import os
import pickle
import time
from contextlib import contextmanager
from typing import List, NoReturn, Optional, Tuple, Union

import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForMaskedLM
from datasets import Dataset, concatenate_datasets, load_from_disk
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

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


class LearnedSparseRetrieval:
    def __init__(
        self,
        embedding_method: str,
        aggregation_method: str,
        similarity_metric: str,
        topk: int,
        retrieval_data_path: Optional[str] = "../../data/wikipedia_documents.json",
        eval_data_path: Optional[str] = "../../data/train_dataset",
    ) -> NoReturn:

        # Validate embedding method
        if embedding_method != "bge-m3":
            raise ValueError(f"Unsupported embedding method: {embedding_method}")

        self.embedding_method = embedding_method
        self.aggregation_method = aggregation_method  # 'sum', 'max'
        self.similarity_metric = similarity_metric  # 'cosine', 'dot_product'
        self.topk = topk

        if embedding_method == "bge-m3":
            embedding_model_name = "upskyy/bge-m3-korean"

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        self.encoder = AutoModelForMaskedLM.from_pretrained(embedding_model_name)
        self.encoder.eval()
        self.encoder.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)

        self.retrieval_data_path = retrieval_data_path
        self.eval_data_path = eval_data_path

        self.contexts = None
        self.p_embedding = None

    # load 'wikipedia_documents.json' data
    def load_data(self):
        logger.info("Loading Wikipedia data from %s", self.retrieval_data_path)
        if os.path.isfile(self.retrieval_data_path):
            file_size = sizeof_fmt(os.path.getsize(self.retrieval_data_path))
            logger.info("File size of %s: %s", self.retrieval_data_path, file_size)
        else:
            logger.warning("File %s does not exist.", self.retrieval_data_path)

        with open(self.retrieval_data_path, "r", encoding="utf-8") as f:
            wiki = json.load(f)

        # unique text 추출
        wiki_df = pd.DataFrame(wiki.values())
        wiki_unique_df = wiki_df.drop_duplicates(subset=["text"], keep="first")
        self.contexts = wiki_unique_df["text"].tolist()
        logger.info(f"Length of unique context: {len(self.contexts)}")

    # Generate sparse embedding for contexts (methods: BGE-M3)
    def get_sparse_embedding(self, batch_size: int = 8) -> NoReturn:
        pickle_name = f"{self.embedding_method}_sparse_embedding_agg-{self.aggregation_method}_similarity-{self.similarity_metric}.bin"
        emb_path = os.path.join(pickle_name)

        if os.path.isfile(emb_path):
            logger.info("Loading BGE-M3 pickle file.")
            logger.info(
                "File size of %s: %s", emb_path, sizeof_fmt(os.path.getsize(emb_path))
            )
            with open(emb_path, "rb") as file:
                self.p_embedding = pickle.load(file)
        else:
            logger.info("Generating embeddings using %s.", self.embedding_method)
            with torch.no_grad():
                all_embeddings = []
                dataloader = DataLoader(self.contexts, batch_size=batch_size)

                with timer("BGE-M3 Embedding Generation"):
                    for batch in tqdm(dataloader, desc="Generating Passage Embeddings"):
                        encoded_input = self.tokenizer(
                            batch, padding=True, truncation=True, return_tensors="pt"
                        ).to(self.device)

                        with torch.no_grad():
                            model_output = self.encoder(**encoded_input)

                            # Pooling
                            logits = model_output.logits
                            attention_mask = encoded_input["attention_mask"]
                            relu_log = torch.log(1 + torch.relu(logits))
                            weighted_log = relu_log * attention_mask.unsqueeze(-1)

                            if self.aggregation_method == "sum":
                                values = torch.sum(weighted_log, dim=1)
                            elif self.aggregation_method == "max":
                                values, _ = torch.max(weighted_log, dim=1)
                            else:
                                raise ValueError(
                                    f"Unsupported aggregation method: {self.aggregation_method}"
                                )

                            # normalization for cosine similarity
                            if self.similarity_metric == "cosine":
                                eps = 1e-9
                                values = values / (
                                    torch.norm(values, dim=-1, keepdim=True) + eps
                                )

                            batch_embedding = values.squeeze()
                            all_embeddings.append(batch_embedding.cpu().numpy())

                        # GPU 캐시 삭제
                        del (
                            encoded_input,
                            model_output,
                            weighted_log,
                            relu_log,
                            values,
                            batch_embedding,
                        )
                        torch.cuda.empty_cache()

                    self.p_embedding = np.vstack(all_embeddings).astype(np.float32)
                    logger.info(
                        f"Generated passage embeddings shape: {self.p_embedding.shape}"
                    )

            # Save embeddings to file
            logger.info("Saving embeddings to %s", emb_path)
            with open(emb_path, "wb") as file:
                pickle.dump(self.p_embedding, file)
            logger.info(
                "File %s saved with size %s",
                emb_path,
                sizeof_fmt(os.path.getsize(emb_path)),
            )

    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:
        if topk is None:
            topk = self.topk

        assert (
            self.p_embedding is not None
        ), "get_sparse_embedding() 메소드를 먼저 수행해주세요."
        logger.debug("Using BGE-M3 for retrieval.")

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

            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="Sparse retrieval: ")
            ):
                result = {
                    "question": example["question"],
                    "id": example["id"],
                    "context": [
                        self.contexts[pid] for pid in doc_indices[idx]
                    ],  # format: list
                }
                if "context" in example.keys() and "answers" in example.keys():
                    result["original_context"] = example["context"]
                    result["answers"] = example["answers"]
                retrieved_data.append(result)

            logger.info("Completed retrieval for dataset queries.")
            return retrieved_data

    def get_relevant_doc(self, query: str, k: Optional[int] = 1) -> Tuple[List, List]:
        query_vec = self.tokenizer(
            [query], padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            model_output = self.encoder(**query_vec)

            # Pooling
            logits = model_output.logits
            attention_mask = query_vec["attention_mask"]
            relu_log = torch.log(1 + torch.relu(logits))
            weighted_log = relu_log * attention_mask.unsqueeze(-1)

            if self.aggregation_method == "sum":
                values = torch.sum(weighted_log, dim=1)
            elif self.aggregation_method == "max":
                values, _ = torch.max(weighted_log, dim=1)
            else:
                raise ValueError(
                    f"Unsupported aggregation method: {self.aggregation_method}"
                )

            # normalization for cosine similarity
            if self.similarity_metric == "cosine":
                eps = 1e-9
                values = values / (torch.norm(values, dim=-1, keepdim=True) + eps)

            q_embedding = values.squeeze().cpu().numpy().astype(np.float32)

        # Clear GPU cache
        del query_vec, model_output, weighted_log, relu_log, values
        torch.cuda.empty_cache()

        # 연관 문서 찾기
        similarity_scores = np.dot(q_embedding, self.p_embedding.T)
        if not isinstance(similarity_scores, np.ndarray):
            similarity_scores = similarity_scores.toarray()

        sorted_result = np.argsort(similarity_scores.squeeze())[::-1]
        doc_score = similarity_scores.squeeze()[sorted_result].tolist()[:k]
        doc_indices = sorted_result.tolist()[:k]
        return doc_score, doc_indices

    def get_relevant_doc_bulk(
        self, queries: List, k: Optional[int] = 1
    ) -> Tuple[List, List]:
        all_embeddings = []
        batch_size = 64
        dataloader = DataLoader(queries, batch_size=batch_size)

        for batch in tqdm(dataloader, desc="Processing Query Batches"):
            query_vecs = self.tokenizer(
                batch, padding=True, truncation=True, return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                model_output = self.encoder(**query_vecs)

                # Pooling
                logits = model_output.logits
                attention_mask = query_vecs["attention_mask"]
                relu_log = torch.log(1 + torch.relu(logits))
                weighted_log = relu_log * attention_mask.unsqueeze(-1)

                if self.aggregation_method == "sum":
                    values = torch.sum(weighted_log, dim=1)
                elif self.aggregation_method == "max":
                    values, _ = torch.max(weighted_log, dim=1)
                else:
                    raise ValueError(
                        f"Unsupported aggregation method: {self.aggregation_method}"
                    )

                # normalization for cosine similarity
                if self.similarity_metric == "cosine":
                    eps = 1e-9
                    values = values / (torch.norm(values, dim=-1, keepdim=True) + eps)

                batch_q_embedding = values.squeeze().cpu().numpy()
                all_embeddings.append(batch_q_embedding)

            # 불필요한 캐시 삭제
            del (
                query_vecs,
                model_output,
                weighted_log,
                relu_log,
                values,
                batch_q_embedding,
            )
            torch.cuda.empty_cache()

        q_embedding = np.vstack(all_embeddings).astype(np.float32)

        # 연관 문서 찾기
        similarity_scores = np.dot(q_embedding, self.p_embedding.T)
        if not isinstance(similarity_scores, np.ndarray):
            similarity_scores = similarity_scores.toarray()

        doc_scores = []
        doc_indices = []

        for i in range(similarity_scores.shape[0]):
            sorted_result = np.argsort(similarity_scores[i, :])[::-1]
            doc_scores.append(similarity_scores[i, :][sorted_result].tolist()[:k])
            doc_indices.append(sorted_result.tolist()[:k])
        return doc_scores, doc_indices

    def evaluate(self, evaluation_method: str = "hit"):
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
            logger.error(
                "Evaluation data path %s does not exist or is not a directory.",
                self.eval_data_path,
            )
            raise FileNotFoundError(
                f"Evaluation data path {self.eval_data_path} not found."
            )

        result_list = self.retrieve(eval_df, topk=self.topk)
        if evaluation_method == "hit":  # k개의 추천 중 선호 아이템이 있는지 측정
            logger.info("Evaluating Hit@k.")
            hits = 0
            for example in result_list:
                if (
                    "original_context" in example
                    and example["original_context"] in example["context"]
                ):
                    hits += 1
            hit_at_k = hits / len(result_list) if len(result_list) > 0 else 0.0
            logger.info(f"Hit@{self.topk}: {hit_at_k:.4f}")
            return hit_at_k
        elif evaluation_method == "mrr":
            logger.info("Evaluating MRR@k.")
            mrr_total = 0.0
            for example in result_list:
                if "original_context" in example:
                    try:
                        rank = example["context"].index(example["original_context"]) + 1
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
    parser.add_argument(
        "--embedding_method",
        type=str,
        choices=["bge-m3"],
        required=True,
        help="Embedding method to use: 'bge-m3'",
    )
    parser.add_argument(
        "--aggregation_method",
        type=str,
        choices=["sum", "max"],
        required=True,
        help="Aggregation method to use: 'sum', 'max",
    )
    parser.add_argument(
        "--similarity_metric",
        type=str,
        choices=["cosine", "dot_product"],
        required=True,
        help="Similarity calculation method to use: 'cosine', 'dot_product",
    )
    parser.add_argument(
        "--topk", type=int, required=True, help="Number of top documents to retrieve"
    )
    parser.add_argument(
        "--evaluation_methods",
        type=str,
        nargs="+",
        default=["hit", "mrr"],
        help="Evaluation methods to use: 'hit', 'mrr'",
    )
    return parser.parse_args()


def append_to_csv(output_csv: str, row: dict, headers: List[str]):
    file_exists = os.path.isfile(output_csv)
    with open(output_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


if __name__ == "__main__":
    args = parse_arguments()

    # Initialize tokenizer
    logger.info("Starting Sparse Retrieval System.")

    logger.info("Embedding method: %s", args.embedding_method)
    logger.info("Aggregation method: %s", args.aggregation_method)
    logger.info("Similarity metric: %s", args.similarity_metric)
    logger.info("Top-K: %d", args.topk)
    logger.info("Evaluation method: %s", args.evaluation_methods)

    retriever = LearnedSparseRetrieval(
        embedding_method=args.embedding_method,  # 'tfidf','bm25'
        aggregation_method=args.aggregation_method,
        similarity_metric=args.similarity_metric,
        topk=args.topk,
        retrieval_data_path="../../data/wikipedia_documents.json",
        eval_data_path="../../data/train_dataset",
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
        "aggregation_method": args.aggregation_method,
        "similarity_metric": args.similarity_metric,
        "topk": args.topk,
        "total_time_sec": f"{total_time:.3f}",
    }

    for eval_method, score in evaluation_results.items():
        row[f"{eval_method}@k"] = f"{score:.4f}"

    # Append to CSV
    headers = [
        "embedding_method",
        "aggregation_method",
        "similarity_metric",
        "topk",
        "total_time_sec",
    ] + [f"{method}@k" for method in args.evaluation_methods]
    append_to_csv("test_learned_sparse_embedding.csv", row, headers)

    logger.info("Learned Sparse Retrieval System finished execution.")
