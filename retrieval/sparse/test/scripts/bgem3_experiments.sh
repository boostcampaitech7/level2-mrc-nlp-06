#!/bin/bash

# Set up directories
CONFIG_DIR="/data/ephemeral/home/minji/retrieval/sparse/config"
BASE_CONFIG="${CONFIG_DIR}/learned_sparse_retrieval_config_example.json"  # 기본 config 파일 지정
LOG_DIR="/data/ephemeral/home/minji/retrieval/sparse/test/logs"

# Define arrays for different options
models=("BAAI/bge-m3" "upskyy/bge-m3-korean" "dragonkue/bge-reranker-v2-m3-ko")
retrieval_types=("sparse" "dense" "hybrid")
dense_metric_types=("IP" "COSINE" "L2")
topk_values=(5 10 20 30 50 70 100)

# Function to generate collection_name
generate_collection_name() {
    local model_prefix=$1
    local dense_metric_type=$2
    local collection_name="collection_bge_${model_prefix}_${dense_metric_type}"
    echo "$collection_name"
}

# Function to run experiment
run_experiment() {
    local model=$1
    local retrieval_type=$2
    local dense_metric_type=$3
    local topk=$4

    # Extract the full embedding_model_name (with slashes) for config
    embedding_model_name=$model

    # Extract model prefix (the part before the slash) for collection_name
    model_prefix=$(echo "$model" | cut -d'/' -f1)

    # Set bgem3_type to the retrieval_type (sparse, dense, or hybrid)
    bgem3_type=$retrieval_type

    # Generate collection name using model_prefix
    collection_name=$(generate_collection_name "$model_prefix" "$dense_metric_type")

    # Skip dense_metric_type for sparse retrieval
    if [ "$retrieval_type" == "sparse" ]; then
        config_file="${model_prefix}_${retrieval_type}_${topk}"
    else
        config_file="${model_prefix}_${retrieval_type}_${dense_metric_type,,}_${topk}"
    fi

    config_file_path="${CONFIG_DIR}/${config_file}"
    cp $BASE_CONFIG $config_file_path.json
    log_file="${LOG_DIR}/experiment_${model_prefix}_${retrieval_type}_${dense_metric_type}_${topk}.log"

    # Use the base config file as a template and modify it
    jq --arg topk "$topk" \
       --arg bgem3_type "$bgem3_type" \
       --arg dense_metric_type "$dense_metric_type" \
       --arg embedding_model_name "$embedding_model_name" \
       --arg collection_name "$collection_name" \
       '.topk = ($topk | tonumber) |
        .bgem3_type = $bgem3_type |
        .dense_metric_type = $dense_metric_type |
        .embedding_model_name = $embedding_model_name |
        .collection_name = $collection_name' "${BASE_CONFIG}" > "${config_file_path}.json"
    
    echo "Running experiment: $config_file"
    python /data/ephemeral/home/minji/retrieval/sparse/src/run_learned_sparse_retrieval.py --config $config_file > $log_file 2>&1

    # Extract results from log file and append to CSV
    total_time=$(grep "Total time:" $log_file | awk '{print $3}')
    hit_k=$(grep "hit@" $log_file | awk '{print $2}')
    mrr_k=$(grep "mrr@" $log_file | awk '{print $2}')
}

# Main experiment loop
for model in "${models[@]}"; do
    for retrieval_type in "${retrieval_types[@]}"; do
        if [ "$retrieval_type" == "sparse" ]; then
            for topk in "${topk_values[@]}"; do
                run_experiment $model $retrieval_type "IP" $topk
            done
        else
            for dense_metric_type in "${dense_metric_types[@]}"; do
                for topk in "${topk_values[@]}"; do
                    run_experiment $model $retrieval_type $dense_metric_type $topk
                done
            done
        fi
    done
done

echo "All experiments completed. Results saved"
