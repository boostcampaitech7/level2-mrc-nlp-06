import logging
import sys
import os
from transformers import AutoTokenizer
import time
import argparse
import json
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils_sparse_retrieval import load_config, timer, append_to_csv
from sparse_retrieval import SparseRetrieval

from konlpy.tag import Mecab, Hannanum, Kkma, Komoran, Okt
# morphs: 형태소 분석
mecab_tokenizer = Mecab().morphs


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -    %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Sparse Retrieval Script")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the JSON config file."
    )
    config = parser.parse_args()
    args = load_config(config.config)

    logger.info("Embedding method: %s", args.embedding_method)
    logger.info("Top-K: %d", args.topk)
    logger.info("Evaluation method: %s", args.eval_metric)

    # Record total start time
    total_start_time = time.time()

    # Load data
    logger.info("Loading data.")
    with open(args.corpus_data_path, "r", encoding="utf-8") as f:
        wiki = json.load(f)
    wiki_df = pd.DataFrame(wiki.values())
    wiki_unique_df = wiki_df.drop_duplicates(subset=["text"], keep="first")
    contexts = wiki_unique_df["text"].tolist()  # unique text 추출

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model_name, use_fast=True)
    retriever = SparseRetrieval(
        embedding_method=args.embedding_method,  # 'tfidf','bm25'
        tokenizer=tokenizer.tokenize,
        contexts=contexts,
    )

    # Generate embeddings
    logger.info("Generating sparse embeddings.")
    with timer("Generate sparse embeddings"):
        if args.embedding_method == "tfidf":
            retriever.get_sparse_embedding_tfidf()
        elif args.embedding_method == "bm25":
            retriever.get_sparse_embedding_bm25()

    # Evaluate
    logger.info("Evaluating retrieval performance.")
    evaluation_results = {}
    for eval_method in args.eval_metric:
        with timer(f"Evaluation {eval_method}"):
            result = retriever.evaluate(
                eval_data_path=args.eval_data_path,
                topk=args.topk,
                eval_metric=eval_method,
            )
            evaluation_results[eval_method] = result

    # Record total end time
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    logger.info(f"Total execution time: {total_time:.3f} s")

    # Append to CSV
    sparse_path_name = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    append_to_csv(
        f"{sparse_path_name}/outputs/sparse_embedding_test_{args.embedding_method}.csv",
        args,
        total_time,
        evaluation_results,
    )

    logger.info("Sparse Retrieval System finished execution.")


if __name__ == "__main__":
    main()