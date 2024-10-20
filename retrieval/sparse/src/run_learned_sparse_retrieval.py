import logging
import sys
import os
import time
import argparse
import json
import pandas as pd
from datasets import concatenate_datasets, load_from_disk


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils_sparse_retrieval import load_config, timer, append_to_csv_learned
from learned_sparse_retrieval import LearnedSparseRetrieval
from pymilvus import connections

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -    %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)
sparse_path_name = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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
    ids = wiki_unique_df["document_id"].tolist()

    # Initialize tokenizer

    retriever = LearnedSparseRetrieval(
        embedding_method=args.embedding_method,  # 'splade','bge-m3'
        embedding_model_name=args.embedding_model_name,
        contexts=contexts,
        ids=ids,
        collection_name=args.collection_name,
        bgem3_type=args.bgem3_type, #sparse, dense, hybrid
        dense_metric_type = args.dense_metric_type   # L2, COSINE, IP
    )

    # Generate embeddings
    logger.info("Generating sparse embeddings.")
    with timer("Generate sparse embeddings"):
        if args.embedding_method == "splade":
            retriever.get_sparse_embedding_splade()
        elif args.embedding_method == "bge-m3":
            retriever.get_sparse_embedding_bgem3()


    # Retrieve
    logger.info("Loading evaluation dataset from %s", args.eval_data_path)
    org_dataset = load_from_disk(args.eval_data_path)
    eval_df = concatenate_datasets(
        [
            org_dataset["train"].flatten_indices(),
            org_dataset["validation"].flatten_indices(),
        ]
    )
    logger.info("Evaluation dataset loaded with %d examples.", len(eval_df))
    retrieved_output = retriever.retrieve(eval_df, dense_metric_type = args.dense_metric_type, topk = args.topk)
    retrieved_df = retrieved_output[0]
    # output_file_name = f"{sparse_path_name}/outputs/retrieved_df_{args.embedding_method}_topk{args.topk}.csv"
    # retrieved_df.to_csv(output_file_name, index=False)

    # Evaluate
    logger.info("Evaluating retrieval performance.")
    evaluation_results = {}
    for eval_method in args.eval_metric:
        with timer(f"Evaluation {eval_method}"):
            result = retriever.evaluate(
                retrieved_df=retrieved_df,
                topk=args.topk,
                eval_metric=eval_method,
            )
            evaluation_results[eval_method] = result

    # Record total end time
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    logger.info(f"Total execution time: {total_time:.3f} s")

    # Append to CSV
    evaluation_file_name = f"{sparse_path_name}/outputs/sparse_embedding_test_{args.embedding_method}.csv"
    append_to_csv_learned(evaluation_file_name,args,total_time,evaluation_results)

    logger.info("Sparse Retrieval System finished execution.")


if __name__ == "__main__":
    main()