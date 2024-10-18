#!/bin/bash


# 로그 파일을 저장할 디렉토리
LOG_DIR="../logs"

# 기본 config 파일 경로 설정
config_file="sparse_retrieval_config_example"
config_file_path="/data/ephemeral/home/minji/retrieval/sparse/config/$config_file.json"

# CSV 파일에 헤더 추가 (처음 실행 시)
echo "embedding_method,topk,total_time_sec,hit@k,mrr@k" > "../results/test_sparse_embedding.csv"

# topk 값을 5에서 100까지 5 간격으로 실행
for topk in {5..10..5}
do
  echo "Running with topk=$topk"
  LOG_FILE="${LOG_DIR}/experiment_topk${topk}.log"

  # config 파일을 복사하여 topk 값을 변경
  tmp_config_file="config_topk_${topk}"
  tmp_config_file_path="/data/ephemeral/home/minji/retrieval/sparse/config/$tmp_config_file.json"
  cp $config_file_path $tmp_config_file_path
  
  # jq를 사용하여 topk 값을 수정
  jq --arg topk "$topk" '.topk = ($topk | tonumber)' $config_file_path > $tmp_config_file_path
  
  # Python 스크립트 실행
  python ../../src/run_sparse_retrieval.py --config $tmp_config_file > $LOG_FILE 2>&1
    echo "Experiment with topk=$TOPK completed. Logs saved to $LOG_FILE"
done