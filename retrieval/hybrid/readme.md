## Hybrid Retrieval
Sparse retrieval과 dense retrieval의 혼합 사용
- sparse retrieval의 단어 수준의 직접적인 비교와 dense retrieval의 문맥적 의미의 유사성을 이용한 장점 결합

## Requirements
```
jsonargparse==4.33.2
scikit-learn==1.4.0
transformers==4.45.2
tqdm==4.65.0
datasets==2.15.0
pprint
```

## Usage
**Config**
- [dense retrieval config](./config/dense_retrieval.json)
 ```
  {
  "wiki_path": "../../../../../data/wikipedia_documents.json", # path to wikipedia documents
  "data_path": "../../../../../data/", # path to train/validation dataset
    "train_folder": "train_dataset/",
    "eval_folder": "eval_dataset/",
  "model_name_or_path": "klue/bert-base", # path to model
  "train": true, # train mode on/off
    "train_samples": false, # train with some samples on/off
    "num_samples": 10,
    "batch_size": 8,
    "num_neg": 2,
    "epochs": 5,
    "learning_rate": 2e-5,
    "weight_decay": 1e-2,
  "evaluate": true, # eval mode on/off
  "topk" : 1, # extract top k documents
  "test_ks" : [1, 5, 10, 15, 20] # test topk list
  }
  ```

- [hybrid retrieval config](./config/hybrid_retrieval.json)
 ```
  {
  "wiki_path": "../../../../../data/wikipedia_documents.json", # path to wikipedia documents
  "data_path": "../../../../../data/", # path to train/validation dataset
    "train_folder": "train_dataset/",
    "eval_folder": "eval_dataset/",
  "model_name_or_path": "../model/klue/bert-base", # path to model
  "retriever_type": "two_stage", # ["sparse", "dense", "hybrid", "two_stage"]
  "alpha" : 0.5, # hybrid weight
  "topk" : 1,
  "test" : true,
  "test_ks" : [1, 5, 10, 15, 20]
  }
  ```

**How To Test**
- 실행은 본 폴더의 하위 폴더인 [src](./src)에서 실행
```python
# 지원하는 retriever_type : ["sparse", "dense", "hybrid", "two_stage"]
python train_dense_retrieval.py # dense retrieval 학습
python run_hybrid_retrieval.py # hybrid retrieval 실행
```

**How To Use**
```python
retriever = HybridRetrieval(
        model_args=config,
        data_path=data_path,
        context_path=wiki_path,
    )

query = "대통령을 포함한 미국의 행정부 견제권을 갖는 국가 기관은?"
queries = ["대통령을 포함한 미국의 행정부 견제권을 갖는 국가 기관은?", "샤이닝 폼을 무엇이라고 칭하기도 하나요?"]
retriever.retrieve(retriever_type=retriever_type, query_or_dataset=query, topk=topk) # single query
retriever.retrieve(retriever_type=retriever_type, query_or_dataset=queries, topk=topk) # multiple queries
```

**결과 저장**
- 학습한 모델들은 `./model/` 폴더에, retrieval 결과는 `./outputs/`폴더에 csv 파일로 저장
```python
retriever.test(retriever_type=retriever_type, query_or_dataset=queries, topk=topk)
```
