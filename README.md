# ODQA ver.2 설명서
### 목적
retriever와 reader에서 서로 다른 전처리를 적용한 데이터셋을 사용할 수 있도록 하기 위함  
주의: 각 task는 각 모듈 내에서 실행하고, 데이터 및 모델을 불러오는 경로는 `/data/ephemeral/home`으로 시작하는 절대 경로로 입력  

### Sparse Embedding  
실행 경로: `./retrieval/sparse/`  
```bash
python src/run_sparse_retrieval --config config/sparse_retrieval_config
```  
결과: `./retrieval/sparse/model/`에 sparse embedding 결과인 `.bin` 파일 생성  
(없으면 생성 후 추론, 있으면 바로 추론)  

### Extraction Based MRC  
실행 경로: `./reader2/`  
```bash
#1 터미널 실행  
python src/run_extraction --config config/extraction_config.json --do_train --do_eval

#2 스크립트 실행
bash train_extraction.sh
```
결과: `./reader2/models/`에 학습이 완료된 모델 저장  

### Inference
실행 경로: `./ODQA2/`
```bash
python inference.py --config config/inference_config.json
```
결과: `./ODQA2/outputs/`에 `nbest_predictions.json`과 `predictions.json` 생성