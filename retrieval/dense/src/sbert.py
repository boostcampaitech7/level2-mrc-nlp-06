import os, sys
import pandas as pd
from tqdm import tqdm
import wandb, random, torch
from datasets import load_from_disk
from torch.utils.data import DataLoader

from sentence_transformers.losses import MegaBatchMarginLoss
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator
from sentence_transformers import (util, losses, CrossEncoder, InputExample,
                                   SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments)

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.sbert_utils import load_corpus, load_datasets


class BiEncoder():
    def __init__(self, data_path, model):
        self.data_path = data_path
        self.model = SentenceTransformer(model)

    def train(self, output_dir='./sbert-bi-encoder', epochs=5, batch_size=32, lr=2e-5, weight_decay=0.01):
        train = load_datasets(self.data_path, 'train')
        valid = load_datasets(self.data_path, 'valid')

        train_retrieval = train.remove_columns([col for col in train.column_names if col not in ['question', 'context']])
        train_retrieval = train_retrieval.rename_columns({'question': 'anchor', 'context': 'positive'})
        valid_retrieval = valid.remove_columns([col for col in valid.column_names if col not in ['question', 'context']])
        valid_retrieval = valid_retrieval.rename_columns({'question': 'anchor', 'context': 'positive'})

        loss = MegaBatchMarginLoss(model=self.model)

        args = SentenceTransformerTrainingArguments(
            # Required parameter:
            output_dir=output_dir,
            # Optional training parameters:
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=lr,
            weight_decay=weight_decay,
            warmup_ratio=0.01,
            fp16=True,  # FP16에서 에러 날 경우 False
            batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
            # Optional tracking/debugging parameters:
            eval_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=500,
            save_total_limit=2,
            logging_steps=100,
            run_name="sbert-5",  # Will be used in W&B if `wandb` is installed
        )

        trainer = SentenceTransformerTrainer(
            model=self.model,
            args=args,
            train_dataset=train_retrieval,
            eval_dataset=valid_retrieval,
            loss=loss
        )

        trainer.train()

    def validate(self, data_type):
        wiki = load_corpus(self.data_path)
        corpus = {str(ids): txt for ids, txt in enumerate(wiki)}

        data = load_datasets(self.data_path, data_type=data_type)
        queries = {}
        for i in range(len(data['question'])):
            queries[str(i)] = data['question'][i]

        relevant_docs = {}
        for i in range(len(data['document_id'])):
            relevant_docs[str(i)] = {str(data['document_id'][i])}

        evaluator = InformationRetrievalEvaluator(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs,
            name="retrieval",
            mrr_at_k=[10, 20, 40, 80, 100], # 
            accuracy_at_k=[10, 20, 40, 80, 100],
            ndcg_at_k=[1],
            precision_recall_at_k=[1],
            map_at_k=[1]
        )
        results = evaluator(self.model)
        
        print(f"Bi-Encoder validation scores: \n{results}")
        
        return results

    def retrieve(self, data_type, stage, topk, save_path=None):
        documents = load_corpus(self.data_path)

        data = load_datasets(self.data_path, data_type)
        queries = data['question']

        print(f"embedding documents & queries: this might take some time\n")
        doc_embs = self.model.encode(documents)
        que_embs = self.model.encode(queries)

        print(f"searching relevant documents\n")
        hits = util.semantic_search(query_embeddings=que_embs, corpus_embeddings=doc_embs, top_k=topk)
        
        if stage == 1: # Bi-Encoder만으로 retrieval 수행
            result_df = pd.DataFrame(data)

            print(f"creating DataFrame")
            if data_type == "train" or data_type == "valid": # train, validation dataset
                result_df = result_df.drop(columns=['title', '__index_level_0__'])
                result_df['original_context'] = result_df['context']

                for i, hit in tqdm(enumerate(hits), desc=f'retrieve {data_type}', total=len(hits)):
                    result_df.loc[i, 'context'] = ' '.join([documents[h['corpus_id']] for h in hit])
            else: # test dataset
                result_df['context'] = None

                for i, hit in tqdm(enumerate(hits), desc=f'retrieve {data_type}', total=len(hits)):
                    result_df.loc[i, 'context'] = ' '.join([documents[h['corpus_id']] for h in hit])
            
            if save_path:
                result_df.to_csv(os.path.join(save_path, 'bi-encoder-result.csv'))
            return result_df
        elif stage == 2: # Bi-Encoder에서 retrieve한 결과를 re-ranking할 수 있도록 하는 형태
            return hits
        else:
            print(f"{stage} for \"stage\" is not appropriate. \"stage\" should be either 1 or 2")

    def save_corpus_embeddings(self, save_path):
        documents = load_corpus(self.data_path)

        print(f"embedding documents\n")
        doc_embs = self.model.encode(documents)
        
        torch.save(doc_embs, os.path.join(save_path, 'embeddings.pt'))


class CrEncoder():
    def __init__(self, data_path, model):
        self.data_path = data_path
        self.model = CrossEncoder(model, num_labels=1)

    def train(self, output_dir='./sbert-cross-encoder', epochs=5, batch_size=32, weight_decay=0.01):
        train = load_datasets(self.data_path, 'train')
        valid = load_datasets(self.data_path, 'valid')

        train_reranker = []
        for i in tqdm(range(train.num_rows), desc='pre-train'):
            train_reranker.append(InputExample(texts=[train['question'][i], train['context'][i]], label=1))
        train_dataloader = DataLoader(train_reranker, shuffle=True, batch_size=batch_size)

        valid_reranker = []
        for i in tqdm(range(valid.num_rows), desc='pre-valid'):
            n_idx = [random.randrange(0, valid.num_rows) for _ in range(4)]
            v = {'query': valid['question'][i],
                'positive': [valid['context'][i]],
                'negative': [valid['context'][n_i] for n_i in n_idx]}
            valid_reranker.append(v)

        evaluator = CERerankingEvaluator(samples=valid_reranker, at_k=10, name='ranker')

        self.model.fit(train_dataloader=train_dataloader,
                       epochs=epochs,
                       output_path=output_dir,
                       
                       weight_decay=weight_decay,
                       evaluator=evaluator,
                       evaluation_steps=500,
                       save_best_model=True)

    def validate(self, data_type, topk):
        data = load_datasets(self.data_path, data_type)

        valid_reranker = []
        for i in tqdm(range(data.num_rows), desc='pre-valid'):
            n_idx = [random.randrange(0, data.num_rows) for _ in range(4)]
            v = {'query': data['question'][i],
                'positive': [data['context'][i]],
                'negative': [data['context'][n_i] for n_i in n_idx]}
            valid_reranker.append(v)

        evaluator = CERerankingEvaluator(samples=valid_reranker, at_k=topk, name='ranker')
        results = evaluator(self.model)
        
        print(f"Cross Encoder validation score(mrr@{topk}): \n{results}")
        
        return results

    def retrieve(self, data_type, topk, save_path=None): # Cross Encoder만으로 retrieval 수행
        documents = load_corpus(self.data_path)

        data = load_datasets(self.data_path, data_type)
        queries = data['question']

        print(f"searching relevant documents: it might take a long time depending on the size of the corpus\n")
        results = []
        for q in tqdm(queries, desc=f'retrieve {data_type}', total=len(queries)):
            result = self.model.rank(q, documents, return_documents=True, top_k=topk)
            results.append(result)
        
        print(f"creating DataFrame\n")
        result_df = pd.DataFrame(data)
        if data_type == 'train' or data_type == 'valid': # train, validation dataset
            result_df = pd.DataFrame(result_df).drop(columns=['title', '__index_level_0__'])
            result_df['original_context'] = result_df['context']
            for i, r in enumerate(results):
                result_df.loc[i, 'context'] = ' '.join([d['text'] for d in r])
        else: # test dataset
            result_df = pd.DataFrame(result_df)
            result_df['context'] = None
            for i, r in enumerate(results):
                result_df.loc[i, 'context'] = ' '.join([d['text'] for d in r])
        
        if save_path:
            result.to_csv(os.path.join(save_path, 'cr-encoder-result.csv'))

        return result_df

    def rerank(self, hits, data_type, topk, save_path=None):
        documents = load_corpus(self.data_path)

        data = load_datasets(self.data_path, data_type)
        queries = data['question']

        print(f"Re-Ranking\n")
        results = []
        for i, hit in tqdm(enumerate(hits), desc='re-rank', total=len(hits)):
            q = queries[i]
            c = [documents[h['corpus_id']] for h in hit]
            result = self.model.rank(q, c, return_documents=True, top_k=topk)
            results.append(result)
        
        print(f"creating DataFrame")
        result_df = pd.DataFrame(data)
        if data_type == 'train' or data_type == 'valid': # train, validation dataset
            result_df = pd.DataFrame(result_df).drop(columns=['title', '__index_level_0__'])
            result_df['original_context'] = result_df['context']
            for i, r in enumerate(results):
                result_df.loc[i, 'context'] = ' '.join([d['text'] for d in r])
        else: # test dataset
            result_df = pd.DataFrame(result_df)
            result_df['context'] = None
            for i, r in enumerate(results):
                result_df.loc[i, 'context'] = ' '.join([d['text'] for d in r])
        
        if save_path:
            result.to_csv(os.path.join(save_path, 'cr-encoder-result.csv'))

        return result_df

    def get_embeddings(self):
        pass
