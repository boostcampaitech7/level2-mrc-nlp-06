from contextlib import contextmanager
import time
import json
from box import Box
import os
import csv


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
