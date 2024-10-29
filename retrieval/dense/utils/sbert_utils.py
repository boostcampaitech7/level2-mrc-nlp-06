import os, json
from datasets import load_from_disk


def load_datasets(path, data_type):
    if data_type == "train":
        data = load_from_disk(os.path.join(path, "train"))
    elif data_type == "valid":
        data = load_from_disk(os.path.join(path, "validation"))
    elif data_type == "test":
        data = load_from_disk(os.path.join(path, "test"))
    else:
        print(
            f'No such type for loading dataset! Use "train", "valid", "test" instead {data_type}.'
        )

    return data


def load_corpus(path):
    with open(
        os.path.join(path, "wikipedia_documents.json"), "r", encoding="utf-8"
    ) as f:
        corpus = json.load(f)

    print(f"loading documents\n")
    documents = [corpus[str(i)]["text"] for i in range(len(corpus))]

    return documents
