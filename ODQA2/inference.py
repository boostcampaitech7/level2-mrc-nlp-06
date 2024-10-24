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
from evaluate import load

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from retrieval.sparse.src.sparse_retrieval import SparseRetrieval

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -    %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# retrieval 점수 평가
def evaluate(
    retrieved_df,
    eval_metric: str = "hit",
):
    # 주어진 메트릭에 따라 평가를 수행
    if eval_metric == "hit":  # k개의 추천 중 선호 아이템이 있는지 측정
        hit_at_k = hit(retrieved_df)
        return hit_at_k

    elif eval_metric == "mrr": # 평균 역 순위 계산
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

    valid_dataset = load_from_disk(retriever_config.eval_data_path)['validation']
    test_dataset = load_from_disk(retriever_config.eval_data_path)['test']
    logger.info("Evaluation dataset loaded with %d examples.", len(test_dataset))

    valid_df = retriever.retrieve(valid_dataset, topk=retriever_config.topk, save=False, retrieval_save_path=config.save_path)
    test_df = retriever.retrieve(test_dataset, topk=retriever_config.topk, save=False, retrieval_save_path=config.save_path)
    # result_df.to_csv(os.path.join(config.save_path, 'retrieved.csv'), index=False)

    # Evaluate: Retrieval의 결과 평가 (metric: Hit@K, MRR@K)
    logger.info("Evaluating retrieval performance.")
    evaluation_results = {}
    for eval_method in retriever_config.eval_metric:  # args.eval_metric: 'hit', 'mrr'
        result = evaluate(
            retrieved_df=valid_df,
            eval_metric=eval_method,
        )
        evaluation_results[eval_method] = result

    logger.info("# ------------------------------------------- #")
    logger.info(f"\tRetrieval Score")
    logger.info(f"\tHit@K: {evaluation_results['hit']:.4f}")
    logger.info(f"\tMRR@K: {evaluation_results['mrr']:.4f}")
    logger.info("# ------------------------------------------- #")


    ### MRC
    # with open(os.path.join(reader_config.data_args.dataset_name, 'wikipedia_documents.json'), 'r', encoding='utf-8') as f:
    #     reader_corpus = json.load(f)
    with open(os.path.join(reader_config.data_path, 'wikipedia_documents.json'), 'r', encoding='utf-8') as f:
        reader_corpus = json.load(f)
    valid_df['context'] = None
    test_df['context'] = None
    for idx, doc_ids in enumerate(valid_df['document_id']):
        valid_df.loc[idx, 'context'] = ' '.join([reader_corpus[str(i)]['text'] for i in doc_ids])
    for idx, doc_ids in enumerate(test_df['document_id']):
        test_df.loc[idx, 'context'] = ' '.join([reader_corpus[str(i)]['text'] for i in doc_ids])

    retrieved_validation = Dataset.from_pandas(valid_df)
    retrieved_test = Dataset.from_pandas(test_df)
    # reader = ExtractiveQA(config.reader)
    reader = LlamaQA(config.reader)
    reader.load_model()
    validation_prediction_dict = reader.predict(retrieved_validation, config.save_path)
    test_prediction_dict = reader.predict(retrieved_test, config.save_path)




    # 최종 ODQA 성능 확인 ----------------------------------------------------
    bertscore = load("bertscore")
    em_f1_score = load("squad")

    # score calculation
    bert_predictions = validation_prediction_dict['predicted_answer']
    bert_references = [sample['answers']['text'][0] for sample in valid_dataset]

    ex_predictions = [{'id': sample['id'], 'prediction_text': sample['predicted_answer']} for sample in validation_prediction_dict]
    ex_references = [{'id': sample['id'], 'answers': [{'text': sample['answers']['text'][0], 'answer_start': sample['answers']['answer_start'][0]}]} for sample in valid_dataset]

    trained_validation_bert_score = bertscore.compute(predictions=bert_predictions, references=bert_references, lang="kr", model_type=reader_config.bert_model, device="cuda:0")
    tuned_exact_match_score = em_f1_score.compute(predictions=ex_predictions, references=ex_references)
    tuned_averages = {
        key: sum(trained_validation_bert_score[key])/len(trained_validation_bert_score[key]) for key in ['precision', 'recall', 'f1']
    }
    tuned_averages['exact_match'] = tuned_exact_match_score['exact_match']
    tuned_averages['squad_f1'] = tuned_exact_match_score['f1']
    logger.info(tuned_averages)