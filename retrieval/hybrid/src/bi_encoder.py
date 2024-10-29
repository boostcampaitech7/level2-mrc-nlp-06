import os
import pickle
import time
import random
from tqdm.auto import tqdm, trange
from typing import List, NoReturn, Optional, Tuple, Union

from transformers import BertModel, BertPreTrainedModel
from transformers import AdamW, get_linear_schedule_with_warmup

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import numpy as np

from utils import *

from pprint import pprint


seed = 2024
random.seed(seed)  # python random seed 고정
np.random.seed(seed)  # numpy random seed 고정


class BertEncoder(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):

        outputs = self.bert(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )

        return outputs[1]  # pooled_output


class BiRetrieval:
    def __init__(self, args, dataset, num_neg, tokenizer, q_encoder, p_encoder):

        self.args = args
        self.dataset = dataset
        self.num_neg = num_neg
        self.train_dataloader = None
        self.eval_dataloader = None

        self.optimizer = None
        self.scheduler = None

        self.tokenizer = tokenizer
        self.q_encoder = q_encoder
        self.p_encoder = p_encoder

        self.padding = True  # "max_length"

        self.prepare_in_batch_negative(num_neg=num_neg)

    def prepare_train_dataset(self):
        train_dataset = None
        self.train_dataloader = DataLoader(
            train_dataset, batch_size=self.args.per_device_train_batch_size
        )

    def prepare_eval_dataset(self):
        eval_dataset = None
        self.eval_dataloader = DataLoader(
            eval_dataset, batch_size=self.args.per_device_train_batch_size
        )

    def prepare_in_batch_negative(self, num_neg=2):

        dataset = self.dataset
        tokenizer = self.tokenizer
        padding = self.padding

        corpus = list(set([example for example in dataset["context"]]))
        corpus = np.array(corpus)
        self.corpus = corpus

        p_with_neg = []
        for context in dataset["context"]:
            while True:
                neg_indices = np.random.randint(len(corpus), size=num_neg)

                if context not in corpus[neg_indices]:
                    p_neg = corpus[neg_indices]

                    p_with_neg.append(context)
                    p_with_neg.extend(p_neg)

                    break

        q_seqs = tokenizer(
            dataset["question"], padding=padding, truncation=True, return_tensors="pt"
        )
        p_seqs = tokenizer(
            p_with_neg, padding=padding, truncation=True, return_tensors="pt"
        )

        max_len = p_seqs["input_ids"].size(-1)
        p_seqs["input_ids"] = p_seqs["input_ids"].view(-1, num_neg + 1, max_len)
        p_seqs["attention_mask"] = p_seqs["attention_mask"].view(
            -1, num_neg + 1, max_len
        )
        p_seqs["token_type_ids"] = p_seqs["token_type_ids"].view(
            -1, num_neg + 1, max_len
        )

        train_dataset = TensorDataset(
            q_seqs["input_ids"],
            q_seqs["attention_mask"],
            q_seqs["token_type_ids"],
            p_seqs["input_ids"],
            p_seqs["attention_mask"],
            p_seqs["token_type_ids"],
        )

        self.train_dataloader = DataLoader(
            train_dataset, batch_size=self.args.per_device_train_batch_size
        )

    def set_optimizer_and_scheduler(self):
        args = self.args

        q_encoder = self.q_encoder
        p_encoder = self.p_encoder

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_params = [
            {
                "params": [
                    p
                    for n, p in q_encoder.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in q_encoder.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
            {
                "params": [
                    p
                    for n, p in p_encoder.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in p_encoder.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_params, lr=args.learning_rate, eps=args.adam_epsilon
        )

        t_total = (
            len(self.train_dataloader)
            // args.gradient_accumulation_steps
            * args.num_train_epochs
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
        )

        self.optimizer = optimizer
        self.scheduler = scheduler

    def preprocess_inputs(self, batch):
        args = self.args
        sample_size = self.num_neg + 1
        batch_size = len(batch[0])
        device = args.device

        q_inputs = {
            "input_ids": batch[0].to(device),
            "attention_mask": batch[1].to(device),
            "token_type_ids": batch[2].to(device),
        }

        p_inputs = {
            "input_ids": batch[3].view(batch_size * sample_size, -1).to(device),
            "attention_mask": batch[4].view(batch_size * sample_size, -1).to(device),
            "token_type_ids": batch[5].view(batch_size * sample_size, -1).to(device),
        }

        return q_inputs, p_inputs

    def compute_loss_cos_sim(self, q_outputs, p_outputs, targets):

        batch_size = q_outputs.shape[0]

        q_outputs = q_outputs.view(batch_size, 1, -1)
        p_outputs = p_outputs.view(batch_size, self.num_neg + 1, -1)

        p_positive_outputs = p_outputs[:, :1, :]
        p_negative_outputs = p_outputs[:, 1:, :]

        triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
        loss = triplet_loss(q_outputs, p_positive_outputs, p_negative_outputs)

        return loss

    def compute_loss_nll(self, q_outputs, p_outputs, targets):

        batch_size = q_outputs.shape[0]

        q_outputs = q_outputs.view(batch_size, 1, -1)
        p_outputs = p_outputs.view(batch_size, self.num_neg + 1, -1)

        sim_scores = (q_outputs @ p_outputs.transpose(1, 2)).squeeze()
        sim_scores = sim_scores.view(batch_size, -1)
        sim_scores = F.log_softmax(sim_scores, dim=1)

        loss = F.nll_loss(sim_scores, targets)

        return loss

    def train(self):
        args = self.args
        dataloader = self.train_dataloader

        q_encoder = self.q_encoder
        p_encoder = self.p_encoder

        self.set_optimizer_and_scheduler()
        optimizer, scheduler = self.optimizer, self.scheduler

        # global_step = 0

        q_encoder.train()
        p_encoder.train()

        """
        model.zero_grad() vs optimizer.zero_grad()
        동일한 경우: 모델의 모든 파라미터가 하나의 옵티마이저에 포함되어 있다면, model.zero_grad()와 optimizer.zero_grad()는 동일하게 동작합니다.
        다른 경우: 모델의 파라미터가 여러 옵티마이저에 분산되어 있는 경우, model.zero_grad()를 사용하는 것이 더 안전합니다. 이는 모든 파라미터의 그래디언트를 초기화하기 때문입니다.
        """

        q_encoder.zero_grad()
        p_encoder.zero_grad()
        torch.cuda.empty_cache()

        train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
        for _ in train_iterator:

            # epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            with tqdm(dataloader, unit="batch") as tepoch:
                for batch in tepoch:
                    batch_size = batch[0].shape[0]
                    q_inputs, p_inputs = self.preprocess_inputs(batch)

                    del batch
                    torch.cuda.empty_cache()

                    q_outputs = q_encoder(**q_inputs)
                    p_outputs = p_encoder(**p_inputs)
                    targets = (
                        torch.zeros(batch_size).long().to(args.device)
                    )  # positive example is always first

                    loss = self.compute_loss_nll(q_outputs, p_outputs, targets)
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    q_encoder.zero_grad()
                    p_encoder.zero_grad()

                    del q_inputs, p_inputs
                    torch.cuda.empty_cache()

                    tepoch.set_postfix(loss=f"{str(loss.item())}")
                    # global_step += 1

    def eval(self):
        args = self.args
        dataloader = self.eval_dataloader

        q_encoder = self.q_encoder
        p_encoder = self.p_encoder

        self.set_optimizer_and_scheduler()
        optimizer, scheduler = self.optimizer, self.scheduler

        # global_step = 0

        q_encoder.eval()
        p_encoder.eval()

        with torch.no_grad():

            train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
            for _ in train_iterator:

                # epoch_iterator = tqdm(train_dataloader, desc="Iteration")
                with tqdm(dataloader, unit="batch") as tepoch:
                    for batch in tepoch:
                        batch_size = batch[0].shape[0]
                        q_inputs, p_inputs = self.preprocess_inputs(batch)

                        q_outputs = q_encoder(**q_inputs)
                        p_outputs = p_encoder(**p_inputs)
                        targets = (
                            torch.zeros(batch_size).long().to(args.device)
                        )  # positive example is always first

                        loss = self.compute_loss(q_outputs, p_outputs, targets)

                        tepoch.set_postfix(loss=f"{str(loss.item())}")
                        # global_step += 1

    def from_pretrained(self, model_path):
        self.tokenizer.from_pretrained(f"{model_path}/tokenizer/")
        self.q_encoder.from_pretrained(f"{model_path}/q/")
        self.p_encoder.from_pretrained(f"{model_path}/p/")

    def save_pretrained(self, model_path):
        self.tokenizer.save_pretrained(f"{model_path}/tokenizer/")
        self.q_encoder.save_pretrained(f"{model_path}/q/")
        self.p_encoder.save_pretrained(f"{model_path}/p/")

    def get_relevant_doc(self, query, topk=1):

        device = self.args.device
        tokenizer = self.tokenizer
        padding = self.padding

        q_encoder = self.q_encoder
        p_encoder = self.p_encoder

        q_encoder.to(device)
        p_encoder.to(device)

        q_encoder.eval()
        p_encoder.eval()
        with torch.no_grad():
            if isinstance(query, str):
                query = [query]
            q_seqs_val = tokenizer(
                query, padding=padding, truncation=True, return_tensors="pt"
            ).to(device)
            q_emb = q_encoder(**q_seqs_val)  # .to("cpu")

            p_embs = []
            for p in self.corpus:
                p = tokenizer(
                    p, padding=padding, truncation=True, return_tensors="pt"
                ).to(device)
                p_emb = p_encoder(**p)  # .to("cpu").numpy()
                p_embs += [p_emb]

        # p_embs = torch.Tensor(p_embs).squeze()
        p_emb = torch.cat(p_embs, dim=0)  # .squeeze()

        # e.g. [Q, B_D], [P, B_D] because bertencoder output is [B, B_D]
        dot_product = q_emb @ p_emb.transpose(0, 1)  # [Q, P]
        q_l2_distance = torch.sqrt(torch.sum(q_emb**2, dim=-1)).view(-1, 1)  # [Q, 1]
        p_l2_distance = torch.sqrt(torch.sum(p_emb**2, dim=-1)).view(-1, 1)  # [P, 1]
        cos_sim = dot_product / (
            q_l2_distance @ p_l2_distance.transpose(0, 1)
        )  # [Q, P]
        indices = torch.argsort(cos_sim, dim=-1, descending=True)
        cos_sim = torch.take_along_dim(cos_sim, indices, dim=-1)

        cos_sim = cos_sim.detach().cpu().numpy()
        indices = indices.detach().cpu().numpy()

        return cos_sim[:, :topk].tolist(), indices[:, :topk].tolist()

    def get_relevant_doc_dot_prod(self, query, topk=1):

        device = self.args.device
        tokenizer = self.tokenizer
        padding = self.padding

        q_encoder = self.q_encoder
        p_encoder = self.p_encoder

        q_encoder.to(device)
        p_encoder.to(device)

        q_encoder.eval()
        p_encoder.eval()
        with torch.no_grad():
            if isinstance(query, str):
                query = [query]
            q_seqs_val = tokenizer(
                query, padding=padding, truncation=True, return_tensors="pt"
            ).to(device)
            q_emb = q_encoder(**q_seqs_val)  # .to("cpu")

            p_embs = []
            for p in self.corpus:
                p = tokenizer(
                    p, padding=padding, truncation=True, return_tensors="pt"
                ).to(device)
                p_emb = p_encoder(**p)  # .to("cpu").numpy()
                p_embs += [p_emb]

        # p_embs = torch.Tensor(p_embs).squeze()
        p_embs = torch.cat(p_embs, dim=0)  # .squeeze()

        scores = q_emb @ p_embs.transpose(0, 1)
        indices = torch.argsort(scores, dim=-1, descending=True)
        scores = torch.take_along_dim(scores, indices, dim=-1)

        scores = scores.detach().cpu().numpy()
        indices = indices.detach().cpu().numpy()

        return scores[:, :topk].tolist(), indices[:, :topk].tolist()
