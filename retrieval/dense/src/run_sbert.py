import json
import argparse

from sbert import BiEncoder, CrEncoder

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-c",
        "--config",
        dest="config",
        action="store",
        help="config path for dense passage retrieval",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    if config["encoder_type"] == "bi-encoder":
        model_config = config["model"]
        model = BiEncoder(config["data_path"], model_config["name"])

        if config["task"] == "train":
            model.train(
                output_dir="./models/bi-encoder",
                epochs=model_config["epochs"],
                batch_size=model_config["batch_size"],
                lr=model_config["lr"],
                weight_decay=model_config["weight_decay"],
            )
        elif config["task"] == "test":
            model.retrive(
                data_type=model_config["data_type"],
                stage=1,
                topk=model_config["topk"],
                save_path="./outputs",
            )
    elif config["encoder_type"] == "cr-encoder":
        model_config = config["model"]
        model = CrEncoder(config["data_path"], model_config["name"])

        if config["task"] == "train":
            model.train(
                output_dir="./models/cr-encoder",
                epochs=model_config["epochs"],
                batch_size=model_config["batch_size"],
                weight_decay=model_config["weight_decay"],
            )
        elif config["taks"] == "test":
            model.retrieve(
                data_type=model_config["data_type"],
                topk=model_config["topk"],
                save_path="./outputs",
            )
    elif config["encoder_type"] == "2-stages":
        bi_config = config["bi-encoder"]
        cr_config = config["cr-encoder"]

        retriever = BiEncoder(config["data_path"], bi_config["model"])
        reranker = CrEncoder(config["data_path"], cr_config["model"])

        hits = retriever.retrieve(
            data_type=config["data_type"], stage=2, topk=bi_config["topk"]
        )
        reranker.rerank(
            hits=hits,
            data_type=config["data_type"],
            topk=cr_config["topk"],
            save_path="./outputs",
        )
