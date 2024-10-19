from contextlib import contextmanager
import time
import json
from box import Box
import os
import csv
import torch


@contextmanager
def timer(name):
    t0 = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - t0
        print(f"[{name}] done in {elapsed:.3f} s")


def load_config(config_path: str):
    sparse_path_name = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_config_path = sparse_path_name + "/config/" + config_path + ".json"
    with open(full_config_path, "r") as f:
        config = json.load(f)
        config = Box(config)
    
    return config

# learned sparse embedding pooling 함수
def pooling_fn(encoded_input, model_output, aggregation_method):
    attention_mask = encoded_input["attention_mask"]
    logits = model_output.logits
    relu_log = torch.log(1 + torch.relu(logits))
    weighted_log = relu_log * attention_mask.unsqueeze(-1)

    if aggregation_method == 'sum':
        values = torch.sum(weighted_log, dim=1)
    elif aggregation_method == 'max':
        values, _ = torch.max(weighted_log, dim=1)
    else:
        raise ValueError(f"Unsupported aggregation method: {aggregation_method}")
    return values

# retrieval 평가 점수
def hit(df):
    hits = len(df[df["rank"] != 0])
    hit_at_k = hits / len(df)
    return hit_at_k


def mrr(df):
    mrr_total = 0.0
    df_with_rank = df[df["rank"] != 0]
    for idx, row in df_with_rank.iterrows():
        mrr_total += 1.0 / row["rank"]
    mrr_at_k = mrr_total / len(df)
    return mrr_at_k


def append_to_csv(output_csv: str, args, total_time, evaluation_results):
    row = {
        "embedding_method": args.embedding_method,
        "topk": args.topk,
        "total_time_sec": f"{total_time:.3f}",
    }
    for eval_method, score in evaluation_results.items():
        row[f"{eval_method}@k"] = f"{score:.4f}"
    headers = ["embedding_method", "topk", "total_time_sec"] + [
        f"{method}@k" for method in args.eval_metric
    ]

    file_exists = os.path.isfile(output_csv)
    with open(output_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

def append_to_csv_learned(output_csv: str, args, total_time, evaluation_results):
    row = {
        "embedding_method": args.embedding_method,
        "bgem3_type": args.bgem3_type,
        "dense_metric_type": args.dense_metric_type,
        "embedding_model_name": args.embedding_model_name,
        "collection_name":args.embedding_model_name,
        "topk": args.topk,
        "total_time_sec": f"{total_time:.3f}",
    }
    for eval_method, score in evaluation_results.items():
        row[f"{eval_method}@k"] = f"{score:.4f}"
    headers = ["embedding_method", "topk", "total_time_sec"] + [
        f"{method}@k" for method in args.eval_metric
    ]

    file_exists = os.path.isfile(output_csv)
    with open(output_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
