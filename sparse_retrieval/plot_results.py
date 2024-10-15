import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_results(csv_path: str, output_dir: str = "./plots"):
    # CSV 파일 읽기
    df = pd.read_csv(csv_path)

    # 시각화를 위한 설정
    sns.set(style="whitegrid")

    # Plot Hit@k
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="topk", y="hit@k", hue="embedding_method", marker="o")
    plt.title("Hit@k vs Topk for TF-IDF and BM25")
    plt.xlabel("Top-K")
    plt.ylabel("Hit@k")
    plt.legend(title="Embedding Method")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/hit_at_k_vs_topk.png")
    plt.close()

    # Plot MRR@k
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="topk", y="mrr@k", hue="embedding_method", marker="o")
    plt.title("MRR@k vs Topk for TF-IDF and BM25")
    plt.xlabel("Top-K")
    plt.ylabel("MRR@k")
    plt.legend(title="Embedding Method")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/mrr_at_k_vs_topk.png")
    plt.close()

    # Combined Plot
    fig, axes = plt.subplots(1, 2, figsize=(20, 6))

    sns.lineplot(
        data=df, x="topk", y="hit@k", hue="embedding_method", marker="o", ax=axes[0]
    )
    axes[0].set_title("Hit@k vs Topk")
    axes[0].set_xlabel("Top-K")
    axes[0].set_ylabel("Hit@k")

    sns.lineplot(
        data=df, x="topk", y="mrr@k", hue="embedding_method", marker="o", ax=axes[1]
    )
    axes[1].set_title("MRR@k vs Topk")
    axes[1].set_xlabel("Top-K")
    axes[1].set_ylabel("MRR@k")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/combined_metrics_vs_topk.png")
    plt.close()

    print(f"Plots saved to {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot retrieval performance metrics")
    parser.add_argument(
        "--csv_path",
        type=str,
        default="test_sparse_embedding.csv",
        help="Path to the results CSV file",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./plots", help="Directory to save the plots"
    )
    args = parser.parse_args()

    plot_results(args.csv_path, args.output_dir)
