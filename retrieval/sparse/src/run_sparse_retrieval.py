import logging
import sys
import os
from transformers import AutoTokenizer
import time
import argparse
import json
import pandas as pd
from datasets import load_from_disk, concatenate_datasets

# 현재 파일 기준 2단계 상위 경로를 PATH에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils_sparse_retrieval import load_config, timer, evaluate, append_to_csv, append_to_csv_learned
from sparse_retrieval import SparseRetrieval, LearnedSparseRetrieval
from konlpy.tag import Mecab

# logger 지정
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -    %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main():

    # config 파일의 argument 가져오기
    parser = argparse.ArgumentParser(description="Sparse Retrieval Script")
    parser.add_argument("--config", type=str, required=True, help="Path to the JSON config file.")
    config = parser.parse_args()
    args = load_config(config.config)


    # 실행하는 모델 옵션 확인
    logger.info("# ------------------------------------------- #")
    logger.info("\tEmbedding method: %s", args.embedding_method)
    logger.info("\tTop-K: %d", args.topk)
    logger.info("# ------------------------------------------- #")


    # 시작 시간 기록 (전체 소요 시간 측정에 사용)
    total_start_time = time.time()


    # wikipedia 데이터 불러오기 (corpus)
    logger.info("Loading wikipedia dataset from %s", args.corpus_data_path)
    with open(args.corpus_data_path, "r", encoding="utf-8") as f:
        wiki = json.load(f)
    wiki_df = pd.DataFrame(wiki.values())
    wiki_unique_df = wiki_df.drop_duplicates(subset=["text"], keep="first")  # unique한 text만 남기기
    contexts = wiki_unique_df["text"].tolist()    # unique text 추출
    ids = wiki_unique_df["document_id"].tolist()  # 대응되는 document_id 추출


    # evaluation 데이터 불러오기 (성능 측정에 사용: Hit@K, MRR@K)
    logger.info("Loading evaluation dataset from %s", args.eval_data_path)
    org_dataset = load_from_disk(args.eval_data_path)
    eval_df = concatenate_datasets(
        [
            org_dataset["train"].flatten_indices(),
            org_dataset["validation"].flatten_indices(),
        ]
    )
    logger.info("Evaluation dataset loaded with %d examples.", len(eval_df))


    # 클래스 선언
    if args.embedding_method in ['tfidf', 'bm25']:
        # 토크나이저 선택: konlpy.tag.Mecab 또는 transformers.AutoTokenizer 둘 중 한가지 선택
        if args.tokenizer_model_name == "mecab":
            tokenizer = Mecab().morphs
        else:
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model_name, use_fast=True)
            tokenizer = tokenizer.tokenize
        
        # 클래스 선언 - TF-IDF / BM25
        sparse_retriever = SparseRetrieval(
            embedding_method=args.embedding_method, # 'tfidf', 'bm25'
            tokenizer=tokenizer,                    # 'mecab', 허깅페이스 토크나이저 (예: 'klue/bert-base' 등)
            contexts=contexts,                      # wikipedia contexts
            document_ids=ids                        # wikipedia document_ids
        )
    elif args.embedding_method in ['splade', 'bge-m3']:
        # 클래스 선언 - SPLADE / BGE-M3
        learned_sparse_retriever = LearnedSparseRetrieval(
            embedding_method=args.embedding_method,         # 'splade', 'bge-m3'
            embedding_model_name=args.embedding_model_name, # 허깅페이스 모델 (예: 'BAAI/bge-m3')
            contexts=contexts,                              # wikipedia contexts
            ids=ids,                                        # wikipedia document_ids
            collection_name=args.collection_name,           # Milvus DB에 저장될 collection 이름 (예: 'collection_bge_BAAI_IP')
            bgem3_type=args.bgem3_type,                     # 'sparse', 'dense', 'hybrid'
            dense_metric_type = args.dense_metric_type      # 'L2', 'COSINE', 'IP'
        )


    # Embedding: wikipedia context에 대한 임베딩 생성 (없으면 생성, 있으면 불러옴)
    logger.info("Generating sparse embeddings.")
    with timer("Generate sparse embeddings"):
        if args.embedding_method == "tfidf":
            sparse_retriever.get_sparse_embedding_tfidf()
        elif args.embedding_method == "bm25":
            sparse_retriever.get_sparse_embedding_bm25()
        elif args.embedding_method == "splade":
            learned_sparse_retriever.get_sparse_embedding_splade()
        elif args.embedding_method == "bge-m3":
            learned_sparse_retriever.get_sparse_embedding_bgem3()


    # Retrieval: query에 대한 topk개 유사 문서 반환 (output: 데이터프레임 형식)
    if args.embedding_method in ['tfidf', 'bm25']:
        retrieved_df = sparse_retriever.retrieve(eval_df, topk=args.topk)
    elif args.embedding_method in ['splade', 'bge-m3']:
        retrieved_df = learned_sparse_retriever.retrieve(eval_df, dense_metric_type = args.dense_metric_type, topk = args.topk)


    # Evaluate: Retrieval의 결과 평가 (metric: Hit@K, MRR@K)
    logger.info("Evaluating retrieval performance.")
    evaluation_results = {}
    for eval_method in args.eval_metric:  # args.eval_metric: 'hit', 'mrr'
        with timer(f"Evaluation {eval_method}"):
            result = evaluate(
                retrieved_df=retrieved_df,
                eval_metric=eval_method,
            )
            evaluation_results[eval_method] = result


    # 최종 결과 출력
    total_end_time = time.time()
    total_time = total_end_time - total_start_time  # 총 소요 시간
    logger.info("# ------------------------------------------- #")
    logger.info(f"\tTotal execution time: {total_time:.3f} s")
    logger.info(f"\tHit@K: {evaluation_results['hit']:.4f}")
    logger.info(f"\tMRR@K: {evaluation_results['mrr']:.4f}")
    logger.info("# ------------------------------------------- #")


    # CSV 파일로 결과 저장 (옵션별 결과 비교 용이하게 하기 위함)
    sparse_path_name = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    evaluation_file_name = f"{sparse_path_name}/outputs/{args.embedding_method}_test_results.csv"
    if args.embedding_method in ['tfidf','bge-m3']:
        append_to_csv(evaluation_file_name, args,total_time,evaluation_results,)
    elif args.embedding_method in ['splade', 'bge-m3']:
        append_to_csv_learned(evaluation_file_name,args,total_time,evaluation_results)

    logger.info("Sparse Retrieval System finished execution.")



if __name__ == "__main__":
    main()