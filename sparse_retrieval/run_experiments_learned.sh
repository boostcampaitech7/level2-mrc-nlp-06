#!/bin/bash

# 로그 파일을 저장할 디렉토리
LOG_DIR="logs_learned"

# CSV 파일에 헤더 추가 (처음 실행 시)
echo "embedding_method,aggregation_method,similarity_metric,topk,total_time_sec,hit@k,mrr@k" > "test_learned_sparse_embedding.csv"


# topk 값을 5부터 100까지 5 단위로 설정
for TOPK in {5..100..5}
do
    # aggregation_method를 sum과 max로 설정
    for AGG in sum max
    do
        # similarity_metric을 cosine과 dot_product로 설정
        for SIM in cosine dot_product
        do
            echo "Running experiment with aggregation_method=$AGG, similarity_metric=$SIM, and topk=$TOPK"

            # 로그 파일 이름 설정 (aggregation_method와 similarity_metric 포함)
            LOG_FILE="${LOG_DIR}/experiment_agg-${AGG}_sim-${SIM}_topk${TOPK}.log"

            python sparse_embedding_learned.py \
                --embedding_method bge-m3 \
                --aggregation_method "$AGG" \
                --similarity_metric "$SIM" \
                --topk "$TOPK" \
                --evaluation_methods hit mrr \
                > "$LOG_FILE" 2>&1

            echo "Experiment with aggregation_method=$AGG, similarity_metric=$SIM, and topk=$TOPK completed. Logs saved to $LOG_FILE"
        done
    done
done
