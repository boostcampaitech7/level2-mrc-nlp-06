## Sparse Passage Embedding 
- Sparse Passage Embedding을 활용한 문서 검색을 실행할 수 있는 코드입니다.
- 전통적인 통계 기반 방법인 TF-IDF, BM25를 이용할 수 있습니다 (`SparseRetriever` 클래스)
- 고도화된 임베딩 기법이 적용된 learned sparse retriever 모델인 SPLADE, BGE-M3을 이용할 수 있습니다 (`LearnedSparseRetriever` 클래스).

### 모델 설명
- **TF-IDF (Term Frequency-Inverse Document Frequency)**
    - 문서 내에서 자주 등장하는 단어와 전체 말뭉치에서의 희소성을 바탕으로 단어의 가중치를 계산해 임베딩합니다.
    - TF-IDF는 단순하고 빠른 검색 성능을 제공합니다.
- **BM25**
    - BM25는 TF-IDF를 개선한 알고리즘으로, 문서 내 단어 빈도가 높아질수록 해당 단어의 가중치 증가가 감소하는 형태로 작동합니다.
    - 더 정교한 검색 결과를 제공하는 점이 특징입니다.
- **SPLADE (Sparse Lexical and Dense Embedding)**
    - 문서에서 더 의미 있는 표현을 추출하는 방법으로 sparse 방식이지만, dense 방식의 강점을 혼합하여 성능을 최적화합니다.
- **BGE-M3 (Bidirectional Global Embedding with Mixed-Mode Models)**
    - 문서 임베딩을 sparse, dense, hybrid 방식으로 수행할 수 있으며, 다양한 검색 시나리오에서 사용할 수 있습니다.

### 본 코드에서의 SPLADE 및 BGE-M3
- SPLADE와 BGE-M3는 큰 데이터 크기와 복잡한 계산 과정 때문에 Milvus DB를 사용하여 용량 및 시간 효율성을 극대화하였습니다.

## 폴더 구조
```text
.
|-- README.md
|-- config
|   |-- config_bgem3.json
|   |-- config_bm25.json
|   |-- config_splade.json
|   `-- config_tfidf.json
|-- model
|   |-- bm25.bin
|   |-- tfidf.bin
|   |-- tfidf_sparse_embedding.bin
|   `-- wiki_embedding.db
|-- outputs
|   |-- tfidf.csv
|   `-- bm25.csv
|-- src
|   |-- run_sparse_retrieval.py
|   `-- sparse_retrieval.py
`-- utils
    `-- utils_sparse_retrieval.py
```

## How To
### `run_sparse_retrieval.py` 사용법
기본 경로는 `./retrieval/sparse`로 설정합니다. 다음 명령어를 통해 실행할 수 있습니다.
```bash
python src/run_sparse_retrieval.py --config {config_file_name}
```


### config 예시
- [config](./config/) 폴더에 각 모델별 config 예시 파일이 저장되어 있습니다. 구체적인 내용은 다음과 같습니다.
- [config_tfidf.json](./config/config_tfidf.json) & [config_bm25.json](./config/config_bm25.json)
    - `corpus_data_path`: wikipedia corpus가 저장된 경로
    - `embedding_method`: 임베딩 방법 (`tfidf`, `bm25`)
    - `topk`: 반환할 문서의 수
    - `tokenizer_model_name`: 토크나이저 모델 이름 (허깅페이스 모델명 또는 `mecab`)
    - `eval_data_path`: 성능 평가에 사용할 데이터 경로
    - `eval_metric`: 성능 평가 메트릭 (`hit`, `mrr`)
- [config_splade.json](./config/config_splade.json) & [config_bgem3.json](./config/config_bgem3.json)
    - `corpus_data_path`: wikipedia corpus가 저장된 경로
    - `embedding_method`: 임베딩 방법 (`splade`, `bge-m3`)
    - `bgem3_type`: BGE-M3의 Retrieval 유형 (`sparse`, `dense`, `hybrid`)
    - `dense_metric_type`: Dense 임베딩에 사용할 메트릭 유형. (`L2`, `COSINE`, `IP`)
    - `embedding_model_name`: 사용할 허깅페이스 모델 이름.
    - `collection_name`: Milvus Database에서 사용할 컬렉션 이름.
    - `topk`: 반환할 문서의 수
    - `eval_data_path`: 성능 평가에 사용할 데이터 경로
    - `eval_metric`: 성능 평가 메트릭 (`hit`, `mrr`)

### 결과 저장
- 임베딩 벡터는 `./model/`에 저장됩니다.
- retrieval 결과는 `./outputs/`에 CSV 형식으로 저장됩니다.

## Modules
### 클래스 선언
- 사용 예시: [run_sparse_retrieval.py](./src/run_sparse_retrieval.py)
```python
from sparse_retrieval import SparseRetrieval, LearnedSparseRetrieval

# TF-IDF, BM25
retriever = SparseRetrieval(
    embedding_method=args.embedding_method, # 'tfidf', 'bm25'
    tokenizer=tokenizer,                    # 'mecab', 허깅페이스 토크나이저
    contexts=contexts,                      # wikipedia contexts
)

# SPLADE, BGE-M3
retriever = LearnedSparseRetrieval(
    embedding_method=args.embedding_method,         # 'splade', 'bge-m3'
    embedding_model_name=args.embedding_model_name, # 허깅페이스 모델
    contexts=contexts,                              # wikipedia contexts
    ids=ids,                                        # wikipedia document_ids
    collection_name=args.collection_name,           # Milvus DB에 저장될 collection 명
    bgem3_type=args.bgem3_type,                     # 'sparse', 'dense', 'hybrid'
    dense_metric_type = args.dense_metric_type      # 'L2', 'COSINE', 'IP'
)
```


### Functions in `SparseRetrieval` Class
**`get_sparse_embedding_tfidf()`**, **`get_sparse_embedding_bm25()`**, 
- TF-IDF 임베딩을 생성하고 저장합니다. 
- 이미 임베딩 결과가 저장된 파일이 있으면 이를 불러오고, 없으면 새로 생성

**`retrieve()`**
- query_or_dataset: 쿼리 문자열 또는 evaluation 데이터셋
- topk: default 10
- save: 검색 결과를 CSV 파일로 저장할지 여부. default True
- retrieval_save_path: 검색 결과 저장 경로. default "../outputs/".

**`get_relevant_doc_tfidf()`**, **`get_relevant_doc_bm25()`**, 
- query: 쿼리 문자열
- k: default 10

**`get_relevant_doc_bulk_tfidf()`**, **`get_relevant_doc_bulk_bm25()`**, 
- queries: 쿼리 리스트
- k: default 10


### Functions in `LearnedSparseRetrieval` Class
**`get_sparse_embedding_splade()`**, **`get_sparse_embedding_bgem3()`**
- 임베딩을 생성하고 데이터베이스에 저장
- 이미 임베딩 결과가 저장된 컬렉션이 있으면 이를 불러오고, 없으면 새로 생성

**`retrieve()`**
- query_or_dataset: 쿼리 문자열 또는 evaluation 데이터셋
- dense_metric_type: Dense 임베딩 검색에 사용할 메트릭 유형. default "IP"
- topk: default 10
- save: 검색 결과를 CSV 파일로 저장할지 여부. default True
- retrieval_save_path: 검색 결과 저장 경로. default "../outputs/".

**`dense_search()`**
- query_dense_embedding: 검색할 dense embedding
- dense_metric_type: Dense 임베딩 검색에 사용할 메트릭 유형. default "IP"
- topk: default 10

**`sparse_search()`**
- query_sparse_embedding: 검색할 sparse embedding
- topk: default 10

**`hybrid_search()`**
- query_dense_embedding: 검색할 dense embedding
- query_sparse_embedding: 검색할 sparse embedding
- dense_metric_type: Dense 임베딩 검색에 사용할 메트릭 유형. default "IP"
- sparse_weight: default 1.0
- dense_weight: default 1.0
- topk: default 10

## Reference
- [TF-IDF (scikit-learn)](https://scikit-learn.org/1.5/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- [BM25 (Github source code)](https://github.com/dorianbrown/rank_bm25)
- [Milvus Embeddings](https://milvus.io/docs/ko/embeddings.md)