# Open-Domain Question Answering 프로젝트
> 주어진 Open-Domain 질문에 대해 정확한 답변을 생성하는 모델을 개발하는 프로젝트이다. 
> 
> 네이버 커넥트재단 부스트캠프 AI Tech 7기 NLP 과정의 일환으로써 3주간 진행했다 _(진행기간: 2024.10.02 ~ 2024.10.24)_

## 1. 프로젝트 개요
- 프로젝트 내용 및 목표
- Wrap-Up Report (TBU: 링크 추가 예정)

## 팀 소개

| 이름  | 담당 역할                                                                                                                                           |
| --- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| [서태영](https://github.com/sty0507) | 데이터 전처리(Wikipedia), 데이터 증강 (GPT-4o Prompting)                                                                                                   |
| [오수현](https://github.com/ocean010315) | DPR (SBERT) 구현 및 실험, Generative Reader 실험 (GPT-4o Prompting), Extractive Reader 실험 (KorQuad FineTuning)                                         |
| [이상의](https://github.com/LeSaUi) | 데이터 전처리 (QA Data), Hybrid Retrieval 구현 및 실험                                                                                                     |
| [이정인](https://github.com/leeennn) | 데이터 전처리(Context), 데이터 증강 (AEDA)                                                                                                            |
| [이정휘](https://github.com/LeeJeongHwi) | Inference 구현, Extractive Reader 실험, Ensemble 구현                                                                                                 |
| [정민지](https://github.com/minjijeong98) | Sparse Retrieval 구현 및 실험 (TF-IDF, BM25, BGE-M3, SPLADE), 벡터DB 구축 및 실험, Generative Reader 구현 및 실험(Llama 3.1 Instruction tuning), Github 사용 환경 세팅 |

## 성과
EM(Exact Match) 점수 66.11% 달성 (Baseline 33.06% 대비 33.05%p 개선)

![리더보드 결과](/assets/leaderboard_score.png)
_최종 리더보드 결과 (EM 66.11% 달성)_

## 주요 접근 방법
- 데이터
	- [데이터 분석](./EDA_team_folder/) 및 [전처리](./data_preprocessing/): 개행(`\n`), 마크다운 문법, 외국어 전처리 및 실험
	- [데이터 증강](./data_augmentation/): AEDA, OpenAI GPT-4o 프롬프트 튜닝
- Retrieval model
	- [Sparse](./retrieval/sparse/README.md): TF-IDF, BM25, BGE-M3, SPLADE
	- [Dense](./retrieval/dense/README.md): Bi-Encoder, Cross-Encoder, 2-stage retrieval
	- [Hybrid](./retrieval/hybrid/readme.md): re-rank, hybrid appraoch
- Reader Model 
	- [Extractive](./reader/extractive/README.md): BERT, RoBERTa, KoELECTRA fine-tuning
	- [Abstractive](./reader/abstractive/): GPT-4o Prompt Tuning, Llama-3.1-8B Instruction Tuning (Q-LoRA)
- Ensemble: Hard voting

#### 3. 기술 스택



## 설치 및 사용법
- 사용한 언어와 버전: Python 3.10.15
- 프로젝트 설치하는 방법 (requirements)

모두 최상위 폴더를 기준으로 실행합니다.

#### 1. `requirements.txt`에서 필요한 라이브러리 추가
```python
pip install -r requirements.txt
```

#### 2. Retrieval 모델 학습 / 임베딩 파일 생성
- Sparse: `sparse_train.sh` 실행
    - `sparse_train.sh`에서 `--config` 설정
    - sparse config는 `/retrieval/sparse/config` 에 config 생성 후 사용
    - 실행이 끝나면 `/retrieval/sparse/model` 에 `.bin`파일이 생성 됨
- Dense: `/retrieval/dense/README` 참조

#### 3. Reader 모델 학습
- Extractive: `train_mrc.sh` 실행
    - 필수 인자로 `--type=ext`를 주어야함
    - `train_mrc.sh` 내 `config_name` 수정
        - `config_name`의 경우 `/reader/extractive/config/{filename}`로 전달해야함
        - config 파일 생성 후 datasets, output_dir 경로 설정 필수
    - Evaluation을 진행하는 경우
        - `bash train_mrc.sh --type=ext --do_eval` 로 실행

#### 3. inference 실행
- `inference.sh` 실행
    - `ODQA/config` 파일 설정
        - `qa_config`는 reader 모델의 config 위치로 설정
        - `retrieval_config`는 사용했던 config 파일 이름으로 설정
        - `datasets` 는 데이터셋 경로로 설정
    - Evaluation (Validation case)만 진행하는 경우
        - `bash inference.sh --qa ext --retrieval sparse --do_eval`
    - Predictions를 저장하고 싶은 경우
        - `bash inference.sh --qa ext --retrieval sparse --do_predict`
        - `bash inference.sh --qa ext --retrieval sparse --do_predict --do_valid` (Validation 결과도 같이 저장)


## 협업 방식
#### Github
- issue, pr 템플릿 (링크 추가)
- 커밋 룰
- 브랜치 전략

#### Confluence 
- 예시 이미지 추가


## 4. 소스코드 구조
- TBU
```text
.
├── EDA_team_folder
│   ├── lji_AEDA.ipynb
│   ├── lji_EDA_language.ipynb
│   ├── lji_processing.ipynb
│   ├── lsu_EDA.ipynb
│   ├── minji_EDA.ipynb
│   ├── sty_EDA.ipynb
│   └── tokenizer_eda_jh.ipynb
├── ODQA
│   ├── README.md
│   ├── config
│   │   └── base_config.json
│   ├── ensemble.py
│   ├── inference.py
│   ├── modules
│   │   ├── __init__.py
│   │   ├── denseRetrieval.py
│   │   ├── extractiveQA.py
│   │   └── spraseRetrieval.py
│   └── utils_common.py
├── ODQA2
│   ├── README.md
│   ├── config
│   │   └── inference_config.json
│   ├── extractiveQA.py
│   └── inference.py
├── README.md
├── assets
│   └── leaderboard_score.png
├── data_augmentation
│   └── Prompting
│       └── data_augmentation.ipynb
├── data_preprocessing
│   ├── train_preprocessing.ipynb
│   └── train_preprocessing.py
├── inference.sh
├── reader
│   ├── abstractive
│   │   ├── config
│   │   │   └── base_gen_config.json
│   │   └── src
│   │       ├── __init__.py
│   │       └── train_abs.py
│   ├── extractive
│   │   ├── README.md
│   │   ├── config
│   │   │   ├── base_config.json
│   │   │   └── koelectra.json
│   │   └── src
│   │       ├── __init__.py
│   │       ├── train_ext.py
│   │       └── trainer_ext.py
│   └── utils
│       ├── __init__.py
│       ├── preprocessing.py
│       └── utils_mrc.py
├── retrieval
│   ├── dense
│   │   ├── README.md
│   │   ├── config
│   │   │   ├── 2-stages.json
│   │   │   ├── bi_encoder.json
│   │   │   └── cr_encoder.json
│   │   ├── src
│   │   │   ├── run_sbert.py
│   │   │   └── sbert.py
│   │   ├── test.py
│   │   └── utils
│   │       └── sbert_utils.py
│   ├── hybrid
│   │   ├── config
│   │   │   ├── dense_retrieval.json
│   │   │   └── hybrid_retrieval.json
│   │   ├── readme.md
│   │   ├── src
│   │   │   ├── bi_encoder.py
│   │   │   ├── hybrid_retrieval.py
│   │   │   ├── run_hybrid_retrieval.py
│   │   │   ├── run_hybrid_retrieval_test.py
│   │   │   └── train_dense_retrieval.py
│   │   └── utils
│   │       └── utils.py
│   └── sparse
│       ├── README.md
│       ├── config
│       │   ├── config_bgem3.json
│       │   ├── config_bm25.json
│       │   ├── config_splade.json
│       │   ├── config_tfidf.json
│       ├── src
│       │   ├── run_sparse_retrieval.py
│       │   └── sparse_retrieval.py
│       └── utils
│           └── utils_sparse_retrieval.py
├── sparse_train.sh
├── train_mrc.sh
└── train_preprocessing.ipynb
```



### Reference 
- TBU