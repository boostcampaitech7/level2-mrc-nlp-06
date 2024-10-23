## Hybrid Retrieval
Sparse retrieval과 dense retrieval의 혼합 사용 
- sparse retrieval의 단어 수준의 직접적인 비교와 dense retrieval의 문맥적 의미의 유사성을 이용한 장점 결합

## How To Use
실행은 본 폴더의 하위 폴더인 src에서 실행

```python
# 지원하는 retriever_type : ["sparse", "dense", "hybrid", "two_stage"]
python train_dense_retrieval.py # dense retrieval 학습
python run_hybrid_retrieval.py # hybrid retrieval 실행
```

**결과 저장**  
- 학습한 모델들은 `./model/` 폴더에, retrieval 결과는 `./outputs/`폴더에 csv 파일로 저장  
