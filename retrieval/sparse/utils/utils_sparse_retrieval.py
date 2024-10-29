from contextlib import contextmanager
import time
import json
from box import Box
import os
import csv


@contextmanager
def timer(name):
    t0 = time.time()  # 시작 시간 기록
    try:
        yield  # 코드 실행
    finally:
        elapsed = time.time() - t0  # 경과 시간 계산
        print(f"[{name}] done in {elapsed:.3f} s")  # 경과 시간 출력


# 주어진 경로에서 JSON 형식의 설정 파일을 로드하여 반환
def load_config(config_path: str):
    sparse_path_name = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_config_path = sparse_path_name + "/config/" + config_path + ".json"
    with open(full_config_path, "r") as f:
        config = json.load(f)
        config = Box(config)
    return config


# retrieval 점수 평가
def evaluate(
    retrieved_df,
    eval_metric: str = "hit",
):
    # 주어진 메트릭에 따라 평가를 수행
    if eval_metric == "hit":  # k개의 추천 중 선호 아이템이 있는지 측정
        hit_at_k = hit(retrieved_df)
        return hit_at_k

    elif eval_metric == "mrr":  # 평균 역 순위 계산
        mrr_at_k = mrr(retrieved_df)
        return mrr_at_k


# 선호 아이템이 있는 히트 수를 계산
def hit(df):
    hits = len(df[df["rank"] != 0])
    hit_at_k = hits / len(df)  # 총 데이터 수 대비 비율 계산
    return hit_at_k


# 평균 역 순위(MRR) 계산
def mrr(df):
    mrr_total = 0.0
    df_with_rank = df[df["rank"] != 0]
    for _, row in df_with_rank.iterrows():
        mrr_total += 1.0 / row["rank"]  # 역 순위 합산
    mrr_at_k = mrr_total / len(df)  # 평균 계산
    return mrr_at_k


# 결과를 CSV 파일에 추가 (TF-IDF, BM25)
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

    file_exists = os.path.isfile(output_csv)  # 파일 존재 여부 확인
    with open(output_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:  # 파일이 없으면 헤더 작성
            writer.writeheader()
        writer.writerow(row)  # 결과 추가


# 결과를 CSV 파일에 추가 (SPLADE, BGE-M3)
def append_to_csv_learned(output_csv: str, args, total_time, evaluation_results):
    row = {
        "embedding_method": args.embedding_method,
        "bgem3_type": args.bgem3_type,
        "dense_metric_type": args.dense_metric_type,
        "embedding_model_name": args.embedding_model_name,
        "collection_name": args.collection_name,
        "topk": args.topk,
        "total_time_sec": f"{total_time:.3f}",
    }
    for eval_method, score in evaluation_results.items():
        row[f"{eval_method}@k"] = f"{score:.4f}"
    headers = [
        "embedding_method",
        "bgem3_type",
        "dense_metric_type",
        "embedding_model_name",
        "collection_name",
        "topk",
        "total_time_sec",
    ] + [f"{method}@k" for method in args.eval_metric]

    file_exists = os.path.isfile(output_csv)  # 파일 존재 여부 확인
    with open(output_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:  # 파일이 없으면 헤더 작성
            writer.writeheader()
        writer.writerow(row)  # 결과 추가
