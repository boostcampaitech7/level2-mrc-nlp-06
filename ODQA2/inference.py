import logging
import argparse
import subprocess
from box import Box
import pandas as pd
import os, sys, json
from transformers import AutoTokenizer
from datasets import load_from_disk, Dataset
from extractiveQA import ExtractiveQA
from konlpy.tag import Mecab
from llamaQA import LlamaQA

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from retrieval.sparse.src.sparse_retrieval import SparseRetrieval

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -    %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="./config/inference_config.json")
    args = parser.parse_args()

    ### config 파일들 불러오기
    # retriever, reader의 config 경로가 있는 base config
    with open(args.config, 'r', encoding="utf-8") as f:
        config = json.load(f)
        config = Box(config)

    # retriever config 불러오기
    with open(config.retriever, 'r', encoding='utf-8') as f:
        retriever_config = json.load(f)
        retriever_config = Box(retriever_config)
    
    # reader config 불러오기
    with open(config.reader, 'r', encoding='utf-8') as f:
        reader_config = json.load(f)
        reader_config = Box(reader_config)

    ### Retrieval
    logger.info("Loading data.")
    with open(retriever_config.corpus_data_path, "r", encoding="utf-8") as f:
        wiki = json.load(f)
    wiki_df = pd.DataFrame(wiki.values())
    wiki_unique_df = wiki_df.drop_duplicates(subset=["text"], keep="first")
    contexts = wiki_unique_df["text"].tolist()  # unique text 추출
    document_ids = wiki_unique_df["document_id"].tolist()

    if retriever_config.embedding_method in ['tfidf', 'bm25']:
        if retriever_config.tokenizer_model_name == "mecab":
            tokenizer = Mecab().morphs
        else:
            tokenizer = AutoTokenizer.from_pretrained(retriever_config.tokenizer_model_name, use_fast=True)
            tokenizer = tokenizer.tokenize
    retriever = SparseRetrieval(embedding_method=retriever_config.embedding_method,
                                tokenizer=tokenizer,
                                contexts=contexts,
                                document_ids=document_ids)

    if retriever_config.embedding_method == "tfidf":
        retriever.get_sparse_embedding_tfidf()
    elif retriever_config.embedding_method == "bm25":
        retriever.get_sparse_embedding_bm25()

    test_dataset = load_from_disk(retriever_config.eval_data_path)['test']
    logger.info("Evaluation dataset loaded with %d examples.", len(test_dataset))

    result_df = retriever.retrieve(test_dataset, topk=retriever_config.topk, save=True, retrieval_save_path=config.save_path)
    # result_df.to_csv(os.path.join(config.save_path, 'retrieved.csv'), index=False)

    ### MRC
    with open(os.path.join(reader_config.data_args.dataset_name, 'wikipedia_documents.json'), 'r', encoding='utf-8') as f:
        reader_corpus = json.load(f)
    result_df['context'] = None
    for idx, doc_ids in enumerate(result_df['document_id']):
        result_df.loc[idx, 'context'] = ' '.join([reader_corpus[str(i)]['text'] for i in doc_ids])

    retrieved_test = Dataset.from_pandas(result_df)
    # reader = ExtractiveQA(config.reader)
    reader = LlamaQA(config.reader)
    reader.predict(retrieved_test, config.save_path)