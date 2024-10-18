#!/bin/bash

# 로그 파일을 저장할 디렉토리
LOG_DIR="logs"

# CSV 파일에 헤더 추가 (처음 실행 시)
echo "embedding_method,topk,total_time_sec,hit@k,mrr@k" > "test_sparse_embedding.csv"

# topk 값을 1부터 100까지 5 단위로 설정
for TOPK in {5..100..5}
do
    for EMBEDDING in tfidf bm25
    do
        echo "Running experiment with embedding_method=$EMBEDDING and topk=$TOPK"

        # 로그 파일 이름 설정
        LOG_FILE="${LOG_DIR}/experiment_${EMBEDDING}_topk${TOPK}.log"

        python sparse_embedding.py \
                --embedding_method $EMBEDDING \
                --topk $TOPK \
                --evaluation_methods hit mrr \
                > $LOG_FILE 2>&1

        echo "Experiment with embedding_method=$EMBEDDING and topk=$TOPK completed. Logs saved to $LOG_FILE"
    done
done
