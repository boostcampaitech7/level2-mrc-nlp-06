import os
import sys
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoModel, AdamW, get_linear_schedule_with_warmup

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.utils_retrieval import Tokenizer

class DenseRetrieval():
    def __init__(self, model_name, pool, metric):
        self.model_name = model_name
        self.pool = pool
        self.metric = metric

        self.device = 'cuda' if torch.cuda.is_available else 'cpu'

        self.tokenizer = Tokenizer(self.model_name)

    def preprocess(self, data, is_train):
        if is_train:
            self.tokenizer.question_tokenizer(data)
            self.questions, self.contexts = self.tokenizer.context_tokenizer(data)

            self.dataset = TensorDataset(self.questions['input_ids'].long(), self.questions['attention_mask'].long(), self.questions['token_type_ids'].long(),
                                        self.contexts['input_ids'].long(), self.contexts['attention_mask'].long(), self.contexts['token_type_ids'].long())
        else:
            self.questions = self.tokenizer.question_tokenizer(data)

    def train(self, args, data):
        self.q_model = AutoModel.from_pretrained(self.model_name)
        self.c_model = AutoModel.from_pretrained(self.model_name)

        self.q_model.to(self.device)
        self.c_model.to(self.device)

        self.preprocess(data, is_train=True)
        
        train_dataloader = DataLoader(self.dataset, batch_size=args.per_device_train_batch_size)

        ### bias와 LayerNorm.weights에는 weight decay를 적용하는 것이 의미 없음
        no_decay=['bias', 'LayerNorm.weights']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.q_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in self.q_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
            {'params': [p for n, p in self.c_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in self.c_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        ###

        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

        # Start Training
        global_step = 0
        self.q_model.zero_grad()
        self.c_model.zero_grad()
        torch.cuda.empty_cache()

        print("========== training retrieval ==========")
        for _ in tqdm(range(args.num_train_epochs), desc='epoch', total=args.num_train_epochs, position=0):
            for batch in tqdm(train_dataloader, desc='iteration', position=1, leave=False, total=len(train_dataloader)):
                self.q_model.train()
                self.c_model.train()

                if torch.cuda.is_available():
                    batch = tuple(t.cuda() for t in batch)
                
                q_inputs = {'input_ids': batch[0],
                            'attention_mask': batch[1],
                            'token_type_ids': batch[2]}
                c_inputs = {'input_ids': batch[3],
                            'attention_mask': batch[4],
                            'token_type_ids': batch[5]}
                
                q_outputs = self.q_model(**q_inputs)
                c_outputs = self.c_model(**c_inputs)

                if self.pool:
                    q_outputs = torch.mean(q_outputs['last_hidden_state'], dim=1)
                    c_outputs = torch.mean(c_outputs['last_hidden_state'], dim=1)
                else:
                    # q_outputs = q_outputs['pooler_output']
                    # c_outputs = c_outputs['pooler_output']
                    q_outputs = q_outputs[0][:, 0, :]
                    c_outputs = c_outputs[0][:, 0, :]
            
                if self.metric=='dot':
                    sim_scores = torch.matmul(q_outputs, torch.transpose(c_outputs, 0, 1))
                elif self.metric=='cosine':
                    sim_scores = F.cosine_similarity(q_outputs.unsqueeze(1), c_outputs.unsqueeze(0), dim=2)

                targets = torch.arange(0, sim_scores.size(0)).long()
                if torch.cuda.is_available():
                    targets = targets.to(self.device)
                
                sim_scores = F.log_softmax(sim_scores, dim=1)
                loss = F.nll_loss(sim_scores, targets)
                
                loss.backward()
                optimizer.step()
                scheduler.step()
                self.q_model.zero_grad()
                self.c_model.zero_grad()
                global_step += 1

                torch.cuda.empty_cache()
        
        self.q_model.save_pretrained(os.path.join(args.output_dir, 'q_encoder'))
        self.c_model.save_pretrained(os.path.join(args.output_dir, 'c_encoder'))
        print(f"saved to {args.output_dir}")
    
    def save_embeddings(self, corpus_path, model_path):
        self.c_model = AutoModel.from_pretrained(os.path.join(model_path, 'c_encoder'))
        self.c_model.to(self.device)

        corpus = pd.read_csv(corpus_path)['text']

        with torch.no_grad():
            self.c_model.eval()

            c_embs = []
            for context in tqdm(corpus, desc='embedding corpus', total=len(corpus)):
                context = self.tokenizer.corpus_tokenizer(context).to(self.device)
                if self.pool:
                    c_emb = torch.mean(self.c_model(**context)['last_hidden_state'], dim=1).to('cpu').numpy()
                else:
                    c_emb = self.c_model(**context)['pooler_output'].to('cpu').numpy()
                c_embs.append(c_emb)

        c_embs = torch.Tensor(c_embs).squeeze()  # (num_passage, emb_dim)
        torch.save(c_embs, os.path.join(model_path, 'context_embeddings.pt'))

        print(c_embs.size())

    def validate(self, valid_data, topk, corpus_path=None, model_path=None):
        self.q_model = AutoModel.from_pretrained(os.path.join(model_path, 'q_encoder'))
        self.q_model.to(self.device)

        corpus = pd.read_csv(corpus_path)['text']
        c_embs = torch.load(os.path.join(model_path, 'context_embeddings.pt'))

        with torch.no_grad():
            self.q_model.eval()

            self.preprocess(valid_data, is_train=False)
            self.questions.to(self.device)
            question_dataset = {'input_ids': self.questions['input_ids'],
                                'attention_mask': self.questions['attention_mask'],
                                'token_type_ids': self.questions['token_type_ids']}
            
            if self.pool:
                q_emb = torch.mean(self.q_model(**question_dataset)['last_hidden_state'], dim=1).to('cpu')
            else:    
                q_emb = self.q_model(**question_dataset)['pooler_output'].to('cpu')  #(num_query, emb_dim)

        if self.metric == 'dot':
            sim_scores = torch.matmul(q_emb, torch.transpose(c_embs, 0, 1))
        elif self.metric == 'cosine':
            sim_scores = []
            for i in tqdm(range(0, q_emb.size(0), 100), desc='cosine similarity'):
                q_batch = q_emb[i:i+100]
                sim_batch = F.cosine_similarity(q_batch.unsqueeze(1), c_embs.unsqueeze(0), dim=2)
                sim_scores.append(sim_batch)
            sim_scores = torch.cat(sim_scores, dim=0)
        rank = torch.argsort(sim_scores, dim=1, descending=True).squeeze()
        
        hit = 0
        mrr = 0
        for i in range(sim_scores.shape[0]):
            context = []
            for k in range(topk):
                context.append(corpus[rank[i][k].item()])
            
            if valid_data['context'][i] in context:
                hit += 1
                mrr += 1 / (context.index(valid_data['context'][i]) + 1)

        hit = hit / len(valid_data['question'])
        mrr = mrr / len(valid_data['question'])
        print(f"Hit@k: {hit}, 'MRR@k: {mrr}")
        
        return {'hit': hit, 'mrr': mrr}

    def retrieve(self, test_data, topk, corpus_path=None, model_path=None):
        # question_contexts = pd.DataFrame(columns=['question', 'context', 'scores'])

        self.q_model = AutoModel.from_pretrained(os.path.join(model_path, 'q_encoder'))
        self.q_model.to(self.device)

        corpus = pd.read_csv(corpus_path)['text']
        c_embs = torch.load(os.path.join(model_path, 'context_embeddings.pt'))

        with torch.no_grad():
            self.q_model.eval()

            self.preprocess(test_data, is_train=False)
            self.questions.to(self.device)
            question_dataset = {'input_ids': self.questions['input_ids'],
                                'attention_mask': self.questions['attention_mask'],
                                'token_type_ids': self.questions['token_type_ids']}
            
            if self.pool:
                q_emb = torch.mean(self.q_model(**question_dataset)['last_hidden_state'], dim=1).to('cpu')
            else:    
                q_emb = self.q_model(**question_dataset)['pooler_output'].to('cpu')  #(num_query, emb_dim)

        if self.metric == 'dot':
            sim_scores = torch.matmul(q_emb, torch.transpose(c_embs, 0, 1))
        elif self.metric == 'cosine':
            sim_scores = []
            for i in tqdm(range(0, q_emb.size(0), 100), desc='cosine similarity'):
                q_batch = q_emb[i:i+100]
                sim_batch = F.cosine_similarity(q_batch.unsqueeze(1), c_embs.unsqueeze(0), dim=2)
                sim_scores.append(sim_batch)
            sim_scores = torch.cat(sim_scores, dim=0)
        rank = torch.argsort(sim_scores, dim=1, descending=True).squeeze()
        
        new_df = pd.DataFrame(test_data)
        # train, validation, test dataset에 따라 서로 다른 형태로 반환해야 함
        if 'context' in test_data.column_names:
            new_df = new_df.drop(columns=['__index_level_0__', 'title', 'document_id'])
            new_df['original_context'] = new_df['context']
        else:
            new_df = pd.DataFrame(test_data)
            new_df['context'] = ''
        
        for i in tqdm(range(sim_scores.shape[0]), desc='Q&A pair'):
            context = []
            for k in range(topk):
                context.append(corpus[rank[i][k].item()])
            new_df.loc[i, 'context'] = ' '.join(context)

        # new_df.to_csv(os.path.join(model_path, 'new_context.csv'))

        return new_df 
