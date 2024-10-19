#!/bin/bash

# 기본 config 파일 경로 설정
config_file="bge_upskyy_sparse_10"
LOG_FILE="$../logs/$config_file.log"
python ../../src/learned_run_sparse_retrieval.py --config $config_file > $LOG_FILE 2>&1
echo "Experiment with $config_file completed. Logs saved to $LOG_FILE"

config_file="splade_naver_sparse_10"
LOG_FILE="$../logs/$config_file.log"
python ../../src/learned_run_sparse_retrieval.py --config $config_file > $LOG_FILE 2>&1
echo "Experiment with $config_file completed. Logs saved to $LOG_FILE"