## Dense Passage Embedding
텍스트 데이터를 각 토큰의 의미를 반영하여 dense vector로 임베딩하는 방법  
- 밀집 벡터이기 때문에 추론 시 공간적, 시간적 효율이 높음  
- 텍스트의 의미를 벡터에 표현할 수 있음  

## How To  
### [rub_sbert.py](./src/run_sbert.py) 사용법  
기본 경로는 `./retrieval/dense`로 설정하기

```bash
python src/run_sbert.py --config ./config/{config_file_name}.json
```

**config 예시**  
- [config](./config/) 참고
- [bi_encoder.json](./config/bi_encoder.json) & [cr_encoder.json](./config/cr_encoder.json)
  - `encoder_type`: BiEncoder, CrEncoder 중 사용할 클래스  
  - `data_path`: train, validation, test dataset과 wiki.json이 있는 폴더의 경로  
  - `task`: train, test 중 수행할 작업  
  - `model`: BiEncoder, CrEncoder 각각에서 train과 retrieve 작업에 필요한 변수들  
- [2-stages.json](./config/2-stages.json)  
  - `encoder_type`: 2-stages로 지정하면 BiEncoder를 retreiver로, CrEncoder를 re-ranker로 사용하여 두 단계의 retrieval을 수행하며, 별도의 학습은 수행하지 않음  
  - `bi-encoder`: BiEncoder 인스턴스 생성과 retrieval을 위한 변수들  
  - `cr-encoder`: CrEncoder 인스턴스 생성과 re-rank를 위한 변수들  

**결과 저장**  
- 학습한 모델들은 `./models/`에, retrieval 결과는 `./outputs/`에 csv로 저장  
- BrEncoder를 학습할 때 wandb를 사용하는데, API key가 없다면 건너뛰기  

## 모듈 설명  
### 클래스 선언
- data_path: dataset이 위치한 폴더명  
- model: 모델이 저장된 폴더 또는 hugging face의 모델 uid  
  - BiEncoder는 SentenceTransformer로 불러올 수 있는 모델들  
  - CrossEncoder_는 AutoModel로 불러올 수 있는 모델들  

```python
from sbert import BiEncoder, CrEncoder

DATA_PATH = ""
MODEL_PATH = ""
retriever = BiEncoder(data_path=DATA_PATH, model=MODEL_PATH)
reranker = CrEncoder(data_path=DATA_PATH, model=MODEL_PATH)
```

### BiEncoder 클래스의 함수 설명
**`train`**  
- output_dir: 학습한 모델을 저장할 폴더의 경로(없으면 생성해줌) default: `./sbert-bi-encoder`  
- epoch: default 5
- batch_size: defualt 32
- lr: default 2e-5
- weight_decay: default 0.01

**`validate`**  
- data_type: `train`, `valid` 중 검증할 대상
- topk 개수를 설정하고 싶다면 [sbert.py](./sbert.py)에서 수정하기  

**`retrieve`**  
- data_type: `train`, `valid`, `test` 중 retrieval을 수행할 대상
- stage: retrieval만 수행하려면 1(DataFrame 반환), re-rank로 연결하려면 2(document_id와 score의 dict 반환)
- topk: query 당 상위 몇 개의 문서를 반환할 것인지

**`save_corpus_embeddings`**  
- save_path: corpus의 embedding vector를 pt 파일로 저장할 경로

```python
retriever.train()
retriever.validate('valid')
retriever.retrieve(data_type='test', stage=1, topk=5)
retriever.save_corpus_embeddings('./')
```

### CrEncoder 함수 설명

**`train`**  
- output_dir: 학습한 모델을 저장할 폴더의 경로(없으면 생성해줌) default: `./sbert-cross-encoder`  
- epoch: default 5
- batch_size: defualt 32
- weight_decay: default 0.01
- lr은 CrossEncoder 클래스에서 초기화를 지원하지 않는 듯함

**`validate`**  
- data_type: `train`, `valid` 중 검증할 대상
- topk: query 당 상위 몇 개의 문서를 반환할 것인지  
CrossEncoder 클래스의 rank 함수를 사용하는데, 전체 corpus를 대상으로 하기 때문에 아주 오래 걸림(시도하지 않는 것 추천)  

**`retrieve`**  
CrossEncoder_만으로 retrieval을 수행할 때 사용하는 함수로, DataFrame을 반환  
- data_type: `train`, `valid`, `test` 중 retrieval을 수행할 대상
- topk: query 당 상위 몇 개의 문서를 반환할 것인지

**`rerank`**  
BiEncoder에서 얻은 결과로 re-ranking을 수행할 때 사용하는 함수로, DataFrame을 반환  
- hits: BiEncoder에서 `retrieve(stage=2)`를 수행하고 얻은 결과  
- data_type: `train`, `valid`, `test` 중 reranking을 수행할 대상으로, `retrieve(data_type='')`에서 사용한 값과 일치해야 함  
- topk: query 당 상위 몇 개의 문서를 반환할 것인지를 결정하며, 일반적으로 retrieval에서 100, reranking에서 10으로 설정

**`save_corpus_embeddings`**  
corpus의 embedding vector를 저장하고자 했으나, CrossEncoder에서 지원하는 기능이 아닌 듯하여 추후 방법을 찾으면 업데이트 예정  

```python
reranker.train()
reranker.validate('valid')
reranker.retrieve(data_type='valid', topk=5)

# 2-stages
hits = retriever.retrieve(data_type='valid', stage=2, topk=5)
reranker.rerank(hits, data_type='valid', topk=3)
```

> 참고자료: https://sbert.net/