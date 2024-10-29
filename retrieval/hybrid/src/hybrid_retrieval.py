import json
import os, sys
import pickle
import time
import random
from tqdm.auto import tqdm
from contextlib import contextmanager
from typing import List, NoReturn, Optional, Tuple, Union

import numpy as np
import pandas as pd
from datasets import Dataset, concatenate_datasets, load_from_disk
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModel,
    AutoModelForQuestionAnswering,
)
import torch
from torch.utils.data import DataLoader

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from scipy.sparse import spmatrix

sys.path.append(os.path.abspath("../utils/"))
from utils import *

seed = 2024
random.seed(seed)  # python random seed 고정
np.random.seed(seed)  # numpy random seed 고정


class HybridRetrieval:
    def __init__(
        self,
        model_args=None,
        tokenize_fn=None,
        data_path: Optional[str] = "../data/",
        context_path: Optional[str] = "wikipedia_documents.json",
        document_id=None,
    ) -> NoReturn:
        """
        Arguments:
            tokenize_fn:
                기본 text를 tokenize해주는 함수입니다.
                아래와 같은 함수들을 사용할 수 있습니다.
                - lambda x: x.split(' ')
                - Huggingface Tokenizer
                - konlpy.tag의 Mecab

            data_path:
                데이터가 보관되어 있는 경로입니다.

            context_path:
                Passage들이 묶여있는 파일명입니다.

            data_path/context_path가 존재해야합니다.

        Summary:
            Passage 파일을 불러오고 TfidfVectorizer를 선언하는 기능을 합니다.
        """

        self.model_args = model_args

        self.data_path = data_path
        with open(context_path, "r", encoding="utf-8") as f:
            wiki = json.load(f)

        self.contexts = list(
            dict.fromkeys([v["text"] for v in wiki.values()])
        )  # set 은 매번 순서가 바뀌므로
        print(f"Lengths of unique contexts : {len(self.contexts)}")
        self.ids = list(range(len(self.contexts)))
        self.document_ids = document_id

        self.tokenize_fn = tokenize_fn

        # Transform by vectorizer
        self.tfidfv = None
        self.dimension_reducer = None
        self.alpha = model_args.alpha
        self.bm25 = None

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.q_encoder = None
        self.p_encoder = None
        self.model = None
        self.dense_tokenizer = None
        self.sparse_tokenizer = None
        self.dense_tokenizer = None

        self.p_sparse_embedding = None  # get_passage_sparse_embedding()로 생성합니다
        self.p_dense_embedding = None  # get_passage_dense_embedding()로 생성합니다
        self.p_hybrid_embedding = None  # get_passage_hybrid_embedding()로 생성합니다
        self.indexer = None  # build_faiss()로 생성합니다.

    def set_sparse_tokenizer(self, tokenizer):
        self.sparse_tokenizer = tokenizer

    def set_dense_tokenizer(self, tokenizer):
        self.dense_tokenizer = tokenizer

    def set_dense_encoder(self, q_encoder, p_encoder):
        self.q_encoder = q_encoder
        self.p_encoder = p_encoder

    def set_dense_model(self):
        model_name_or_path = self.model_args.model_name_or_path
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path + "/tokenizer/")
        q_encoder = AutoModel.from_pretrained(model_name_or_path + "/q/").to(
            self.device
        )
        p_encoder = AutoModel.from_pretrained(model_name_or_path + "/p/").to(
            self.device
        )

        self.set_dense_tokenizer(tokenizer)
        self.set_dense_encoder(q_encoder, p_encoder)

    def get_passage_sparse_embedding(self, retriever_type="tfidf") -> NoReturn:
        """
        Summary:
            Passage Embedding을 만들고
            TFIDF와 Embedding을 pickle로 저장합니다.
            만약 미리 저장된 파일이 있으면 저장된 pickle을 불러옵니다.
        """

        model_name = self.model_args.model_name_or_path
        tokenizer_path = model_name
        model_folder_path = (
            "../model/" if model_name.startswith("../model/") else self.data_path
        )

        if tokenizer_path.startswith("../model/"):
            tokenizer_path = tokenizer_path + "/tokenizer/"
        self.sparse_tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path
        )  # use_fast=False

        # Transform by vectorizer
        self.tfidfv = TfidfVectorizer(
            tokenizer=self.sparse_tokenizer.tokenize,
            ngram_range=(1, 2),
            max_features=50000,
        )

        # Pickle을 저장합니다.
        pickle_name = f"{model_name.replace('/', '_').replace('//', '_')}_{retriever_type}_embedding.bin"
        tfidfv_name = (
            f"{model_name.replace('/', '_').replace('//', '_')}_{retriever_type}_v.bin"
        )

        emb_path = os.path.join(model_folder_path, pickle_name)
        tfidfv_path = os.path.join(model_folder_path, tfidfv_name)

        if os.path.isfile(emb_path) and os.path.isfile(tfidfv_path):
            self.p_sparse_embedding = load_pickle(emb_path)
            self.tfidfv = load_pickle(tfidfv_path)
            print("Embedding pickle load.")
        else:
            if retriever_type in ["tfidf", "sparse"]:
                with timer("Build passage embedding"):
                    self.p_sparse_embedding = self.tfidfv.fit_transform(self.contexts)
                    print(
                        "passage sparse embedding shape:", self.p_sparse_embedding.shape
                    )

                save_pickle(self.p_sparse_embedding, emb_path)
                save_pickle(self.tfidfv, tfidfv_path)
                print("Embedding pickle saved.")

        return self.p_sparse_embedding

    def get_passage_dense_embedding(self) -> NoReturn:
        """
        Summary:
            Passage Embedding을 만들고
            Embedding을 pickle로 저장합니다.
            만약 미리 저장된 파일이 있으면 저장된 pickle을 불러옵니다.
        """

        model_args = self.model_args

        # Pickle을 저장합니다.
        model_name = model_args.model_name_or_path
        model_folder_path = (
            "../model/" if model_name.startswith("../model/") else self.data_path
        )

        embedding_name = (
            f"{model_name.replace('/', '_').replace('//', '_')}_bi_dense_embedding.bin"
        )

        emb_path = os.path.join(model_folder_path, embedding_name)
        model_path = os.path.join(model_folder_path, model_name)

        print("model_path", model_path)

        self.set_dense_model()

        if os.path.isfile(emb_path):  # and os.path.isdir(model_path):
            self.p_dense_embedding = load_pickle(emb_path)
            print("Passage Embedding pickle load.")
        else:
            p_encoder = self.p_encoder
            tokenizer = self.dense_tokenizer
            print("Embedding model load.")

            device = self.device
            dataloader = DataLoader(self.contexts, batch_size=32)

            p_dense_embedding_batches = []
            p_dense_embedding = None

            # 모델의 상태 확인
            print(f"모델은 현재 {p_encoder.training} 상태입니다.")
            print(type(p_encoder))

            p_encoder.to(device)
            with timer("passage dense embedding time:"):
                with torch.no_grad():
                    for i, batch in enumerate(dataloader):
                        tokenized_contexts = tokenizer(
                            batch, padding=True, truncation=True, return_tensors="pt"
                        )
                        tokenized_contexts = tokenized_contexts.to(device)

                        output = p_encoder(**tokenized_contexts)
                        if hasattr(output, "pooler_output"):
                            p_dense_embedding = output.pooler_output
                        elif hasattr(output, "hidden_states"):
                            p_dense_embedding = output.hidden_states[-1][:, 0, :]
                        elif hasattr(output, "last_hidden_state"):
                            p_dense_embedding = output.last_hidden_state[:, 0, :]

                        p_dense_embedding_batches += [p_dense_embedding.detach().cpu()]

            self.p_dense_embedding = torch.cat(p_dense_embedding_batches)

            save_pickle(self.p_dense_embedding, emb_path)
            print("Passage Embedding pickle saved.")

        print("Passage dense embedding done.")

        return self.p_dense_embedding

    def get_passage_hybrid_embedding(self):

        model_name = self.model_args.model_name_or_path

        model_folder_path = (
            "../model/" if model_name.startswith("../model/") else self.data_path
        )

        # Pickle을 저장합니다.
        hybrid_name = (
            f"{model_name.replace('/', '_').replace('//', '_')}_hybrid_embedding.bin"
        )
        dimension_reducer_name = (
            f"{model_name.replace('/', '_').replace('//', '_')}_hybridv.bin"
        )

        hybrid_path = os.path.join(model_folder_path, hybrid_name)
        dimension_reducer_path = os.path.join(model_folder_path, dimension_reducer_name)

        if os.path.isfile(dimension_reducer_path):
            self.dimension_reducer = load_pickle(dimension_reducer_path)
            print("Hybrid Embedding pickle load.")

            if self.p_sparse_embedding is None:
                self.get_passage_sparse_embedding()  # no return
            if self.p_dense_embedding is None:
                self.get_passage_dense_embedding()  # no return

            p_sparse_embedding = self.p_sparse_embedding
            p_dense_embedding = self.p_dense_embedding

            with timer("Build passage embedding"):
                if isinstance(p_dense_embedding, torch.Tensor):
                    p_dense_embedding = p_dense_embedding.detach().to("cpu").numpy()
                else:
                    p_dense_embedding = np.array(p_dense_embedding)

                p_sparse_embedding = self.dimension_reducer.transform(
                    p_sparse_embedding
                )

                a = self.alpha
                p_hybrid_embedding = (
                    a * p_sparse_embedding + (1 - a) * p_dense_embedding
                )
                self.p_hybrid_embedding = p_hybrid_embedding

        else:
            if self.p_sparse_embedding is None:
                self.get_passage_sparse_embedding()  # no return
            if self.p_dense_embedding is None:
                self.get_passage_dense_embedding()  # no return

            p_sparse_embedding = self.p_sparse_embedding
            p_dense_embedding = self.p_dense_embedding

            with timer("Build passage embedding"):
                if isinstance(p_dense_embedding, torch.Tensor):
                    p_dense_embedding = p_dense_embedding.detach().cpu().numpy()
                else:
                    p_dense_embedding = np.array(p_dense_embedding)

                # self.dimension_reducer = umap.UMAP(n_components=p_dense_embedding.shape[-1])
                # dimension_reducer = self.dimension_reducer

                # self.dimension_reducer = PCA(n_components=768)
                # dimension_reducer = self.dimension_reducer

                self.dimension_reducer = TruncatedSVD(
                    n_components=768, algorithm="arpack", random_state=seed
                )
                dimension_reducer = self.dimension_reducer

                p_sparse_embedding = dimension_reducer.fit_transform(p_sparse_embedding)

                a = self.alpha
                p_hybrid_embedding = (
                    a * p_sparse_embedding + (1 - a) * p_dense_embedding
                )
                self.p_hybrid_embedding = p_hybrid_embedding

            # print(self.p_hybrid_embedding.shape)

            # save_pickle(dimension_reducer, dimension_reducer_path)
            # print("Hybrid Embedding pickle saved.")
        return self.p_hybrid_embedding

    def retrieve(
        self,
        retriever_type="two_stage",
        query_or_dataset: Union[str, Dataset] = None,
        topk: Optional[int] = 1,
    ) -> Union[Tuple[List, List], pd.DataFrame]:
        """
        Arguments:
            query_or_dataset (Union[str, Dataset]):
                str이나 Dataset으로 이루어진 Query를 받습니다.
                str 형태인 하나의 query만 받으면 `get_relevant_doc`을 통해 유사도를 구합니다.
                Dataset 형태는 query를 포함한 HF.Dataset을 받습니다.
                이 경우 `get_relevant_doc_bulk`를 통해 유사도를 구합니다.
            topk (Optional[int], optional): Defaults to 1.
                상위 몇 개의 passage를 사용할 것인지 지정합니다.

        Returns:
            1개의 Query를 받는 경우  -> Tuple(List, List)
            다수의 Query를 받는 경우 -> pd.DataFrame: [description]

        Note:
            다수의 Query를 받는 경우,
                Ground Truth가 있는 Query (train/valid) -> 기존 Ground Truth Passage를 같이 반환합니다.
                Ground Truth가 없는 Query (test) -> Retrieval한 Passage만 반환합니다.
        """

        if isinstance(query_or_dataset, str):
            query_or_dataset = Dataset.from_dict(
                {"question": [query_or_dataset], "id": [0]}
            )

        if retriever_type not in ["bm25"] and self.p_sparse_embedding is None:
            self.get_passage_sparse_embedding()
            assert (
                self.p_sparse_embedding is not None
            ), "get_passage_sparse_embedding() 메소드를 먼저 수행해줘야합니다."

        if (
            retriever_type not in ["bm25", "sparse"]
        ) and self.p_dense_embedding is None:
            self.get_passage_dense_embedding()
            assert (
                self.p_dense_embedding is not None
            ), "get_passage_dense_embedding() 메소드를 먼저 수행해줘야합니다."

        if retriever_type == "hybrid" and self.p_hybrid_embedding is None:
            self.get_passage_hybrid_embedding()
            assert (
                self.p_hybrid_embedding is not None
            ), "get_passage_hybrid_embedding() 메소드를 먼저 수행해줘야합니다."

        stage1_topk = topk * 10
        stage2_topk = topk

        print(
            "[Search queries]\n",
            ", ".join(query_or_dataset["question"][:3]),
            ", ... etc. \n",
        )

        if retriever_type in [
            "bm25",
            "sparse",
            "dense",
            "hybrid",
            "two_stage",
            "bm25s_two_stage",
            "bm25s_hybrid",
        ]:
            print(f"[Search all queries by {retriever_type}] \n")

            with timer("query exhaustive search"):
                dataloader = DataLoader(query_or_dataset["question"], batch_size=32)
                doc_scores_batches = []
                doc_indices_batches = []

                for i, queries in enumerate(dataloader):
                    if "two_stage" in retriever_type:
                        doc_scores, doc_indices = self.get_relevant_doc(
                            retriever_type="sparse", queries=queries, k=stage1_topk
                        )
                        doc_scores, doc_indices = self.rerank_doc(
                            retriever_type="dense",
                            queries=queries,
                            doc_scores=doc_scores,
                            doc_indices=doc_indices,
                            topk=stage2_topk,
                        )
                    else:
                        doc_scores, doc_indices = self.get_relevant_doc(
                            retriever_type=retriever_type,
                            queries=queries,
                            k=stage2_topk,
                        )

                    doc_scores_batches += doc_scores
                    doc_indices_batches += doc_indices

                doc_scores = doc_scores_batches
                doc_indices = doc_indices_batches

                if len(doc_scores) == 1:
                    for i in range(topk):
                        print(f"Top-{i+1} passage with score {doc_scores[0][i]:4f}")
                        print(self.contexts[doc_indices[0][i]][:100])

                else:
                    total = []
                    for idx, example in enumerate(
                        tqdm(query_or_dataset, desc="Sparse retrieval: ")
                    ):
                        if idx >= len(doc_indices):
                            break

                        docs = [self.contexts[pid] for pid in doc_indices[idx]]
                        document_ids = (
                            [self.document_ids[pid] for pid in doc_indices[idx]]
                            if self.document_ids is not None
                            else None
                        )

                        tmp = {
                            # Query와 해당 id를 반환합니다.
                            "question": example["question"],
                            "id": example["id"],
                            # Retrieve한 Passage의 id, context를 반환합니다.
                            "context": " ".join(docs),
                            "document_id": document_ids,
                            "original_document_rank": (
                                docs.index(example["context"]) + 1
                                if example["context"] in docs
                                else float("inf")
                            ),
                        }
                        if "context" in example.keys() and "answers" in example.keys():
                            # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                            tmp["original_context"] = example["context"]
                            tmp["answers"] = example["answers"]
                        total.append(tmp)

                    cqas = pd.DataFrame(total)
                    return cqas

    def get_relevant_doc(
        self, retriever_type="sparse", queries: List[str] = None, k: Optional[int] = 1
    ) -> Tuple[List, List]:
        """
        Arguments:
            queries (List):
                하나 이상의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """

        sparse_queries_vec = self.tfidfv.transform(queries)

        assert (
            np.sum(sparse_queries_vec) != 0
        ), "Error: The Words of questions not in the vocab of vectorizer"

        if retriever_type in ["dense", "hybrid"]:
            with torch.no_grad():
                tokenized_queries = self.dense_tokenizer(
                    queries, padding=True, truncation=True, return_tensors="pt"
                ).to(self.device)

                self.q_encoder.to(self.device)
                output = self.q_encoder(**tokenized_queries)
                if hasattr(output, "pooler_output"):
                    dense_queries_vec = output.pooler_output
                elif hasattr(output, "hidden_states"):
                    dense_queries_vec = output.hidden_states[-1][:, 0, :]
                elif hasattr(output, "last_hidden_state"):
                    dense_queries_vec = output.last_hidden_state[:, 0, :]

                dense_queries_vec = dense_queries_vec.detach().cpu().numpy()

                assert (
                    np.sum(dense_queries_vec) != 0
                ), "Error: The Words of questions not in the vocab of vectorizer"

        if retriever_type in ["hybrid"]:
            sparse_queries_vec = self.dimension_reducer.transform(sparse_queries_vec)

            a = self.alpha
            hybrid_queries_vec = a * sparse_queries_vec + (1 - a) * dense_queries_vec

            # print("sparse", sparse_queries_vec.shape, type(sparse_queries_vec))
            # print("dense", dense_queries_vec.shape, type(dense_queries_vec))
            # print("hybrid", hybrid_queries_vec.shape, type(hybrid_queries_vec))

        q_embedding = None
        p_embedding = None

        match retriever_type:
            case "sparse":
                q_embedding = sparse_queries_vec
                p_embedding = self.p_sparse_embedding
            case "dense":
                q_embedding = dense_queries_vec
                p_embedding = self.p_dense_embedding
            case "hybrid":
                q_embedding = hybrid_queries_vec
                p_embedding = self.p_hybrid_embedding

        if not isinstance(q_embedding, spmatrix) or not isinstance(
            p_embedding, spmatrix
        ):
            if hasattr(q_embedding, "toarray"):
                q_embedding = q_embedding.toarray()
            if hasattr(p_embedding, "toarray"):
                p_embedding = p_embedding.toarray()

            q_embedding = torch.Tensor(q_embedding).to(self.device)
            p_embedding = torch.Tensor(p_embedding).to(self.device)

        doc_scores = q_embedding @ p_embedding.T

        if isinstance(doc_scores, torch.Tensor):
            doc_scores = doc_scores.detach().cpu().numpy()
        if hasattr(doc_scores, "toarray"):
            doc_scores = doc_scores.toarray()

        doc_indices = np.argsort(doc_scores)[:, ::-1].copy()
        doc_scores = np.take_along_axis(doc_scores, doc_indices, axis=-1)

        return doc_scores[:, :k].tolist(), doc_indices[:, :k].tolist()

    def rerank_doc(
        self,
        retriever_type="sparse",
        queries=None,
        doc_scores=None,
        doc_indices=None,
        topk=1,
    ):

        if type(queries) is str:
            queries = [queries]

        if retriever_type == "sparse":
            with timer("sparse reranker"):
                query_vec = self.tfidfv.transform(queries)
                p_embedding = self.p_sparse_embedding

                query_vec = query_vec.toarray()
                p_embedding = p_embedding.toarray()

        elif retriever_type == "dense":
            with timer("dense reranker"):
                with torch.no_grad():
                    tokenized_queries = self.dense_tokenizer(
                        queries, padding=True, truncation=True, return_tensors="pt"
                    ).to(self.device)

                    self.q_encoder.to(self.device)
                    output = self.q_encoder(**tokenized_queries)
                    if hasattr(output, "pooler_output"):
                        query_vec = output.pooler_output
                    elif hasattr(output, "hidden_states"):
                        query_vec = output.hidden_states[-1][:, 0, :]
                    elif hasattr(output, "last_hidden_state"):
                        query_vec = output.last_hidden_state[:, 0, :]

                    query_vec = query_vec.detach().cpu().numpy()
                    assert (
                        np.sum(query_vec) != 0
                    ), "Error: The Words of questions not in the vocab of vectorizer"

                    p_embedding = self.p_dense_embedding

        query_vec = torch.tensor(query_vec).to(self.device)
        p_embedding = torch.tensor(p_embedding).to(self.device)
        p_embedding = p_embedding[torch.tensor(doc_indices)]

        query_vec = query_vec.unsqueeze(1)

        result = query_vec @ p_embedding.view(
            query_vec.shape[0], -1, p_embedding.shape[-1]
        ).transpose(1, 2)

        if isinstance(result, torch.Tensor):
            result = result.detach().cpu().numpy()
        elif not isinstance(result, np.ndarray):
            result = result.toarray()

        result = result.squeeze(1)
        result_sorted_indices = np.argsort(result)[:, ::-1].copy()
        doc_rerank_scores = np.take_along_axis(result, result_sorted_indices, axis=-1)

        doc_indices = np.array(doc_indices).reshape(result_sorted_indices.shape)
        doc_rerank_indices = np.take_along_axis(
            doc_indices, result_sorted_indices, axis=-1
        )

        return (
            doc_rerank_scores[:, :topk].tolist(),
            doc_rerank_indices[:, :topk].tolist(),
        )

    def test(self, retriever_type="sparse", query_or_dataset=None, topk=1):

        print(
            f"\n@@@@@@@@@@@@@@@@@@@@@@@@@ {retriever_type} method @@@@@@@@@@@@@@@@@@@@@@@@@"
        )

        with timer("exhaustive search"):
            df = self.retrieve(
                retriever_type=retriever_type,
                query_or_dataset=query_or_dataset,
                topk=topk,
            )

            df["hit@k"] = df.apply(
                lambda row: row["original_context"] in row["context"], axis=1
            )
            df["mrr@k"] = df.apply(
                lambda row: 1 / row["original_document_rank"], axis=1
            )

            hit_k = df["hit@k"].sum() / len(df)
            mrr_k = df["mrr@k"].sum() / len(df)

            print(
                f"hit@{topk} {retriever_type} retrieval result by exhaustive search: {hit_k:.4f}"
            )
            print(
                f"mrr@{topk} {retriever_type} retrieval result by exhaustive search: {mrr_k:.4f}"
            )

        # with timer("An example"):
        #    df = self.retrieve(retriever_type=retriever_type, query_or_dataset=query_or_dataset, topk=topk)

        # DataFrame에 결과 추가
        retriever_name = (
            retriever_type + f"_alpha_{self.alpha}"
            if retriever_type == "hybrid"
            else retriever_type
        )

        df = pd.DataFrame(
            {
                "retriever_name": [retriever_name],
                "topk": [topk],
                "hit@k": [hit_k],
                "mrr@k": [mrr_k],
            }
        )

        # 저장할 폴더 경로를 지정
        output_dir = "../outputs/"
        # 폴더가 존재하지 않으면 생성
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        df.to_csv(f"../outputs/output_{retriever_name}_topk_{topk}.csv", index=False)

        print(
            f"@@@@@@@@@@@@@@@@@@@@@@@@@  {retriever_type} done  @@@@@@@@@@@@@@@@@@@@@@@@@\n"
        )
