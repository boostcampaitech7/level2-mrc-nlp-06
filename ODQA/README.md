# ODQA Inference

해당 코드는 ODQA Task를 Inference하는 코드입니다.

## How to

테스트하는 방법은 다음과 같습니다

### 1. Retrieval 학습

* Sparse의 경우 최상위 폴더에서 `sparse_train.sh`를 실행

    * `sparse_train.sh`에서 `--config` 설정

    * sparse config는 `/retrieval/sparse/config` 에 config 생성 후 사용

    * 실행이 끝나면 `/retrieval/sparse/model` 에 `.bin`파일이 생성 되어야함

* Dense의 경우 `/retrieval/dense/README` 참조

### 2. Reader 학습

> 현재 Extractive만 사용 가능

* 최상위 폴더에서 `train_mrc.sh` 실행

    * 필수 인자로 `--type=ext`를 주어야함

    * `train_mrc.sh` 내 `config_name` 수정
    
        * `config_name`의 경우 `/reader/extractive/config/{filename}`로 전달해야함

        * config 파일 생성 후 datasets, output_dir 경로 설정 필수

    * Evaluation을 진행하는 경우
        
        * `bash train_mrc.sh --type=ext --do_eval` 로 실행


### 3. Inference 실행

* 최상위 폴더에서 `inference.sh` 실행
    
    * `ODQA/config` 파일 설정
    
        * `qa_config`는 reader 모델의 config 위치로 설정

        * `retrieval_config`는 사용했던 config 파일 이름으로 설정

        * `datasets` 는 데이터셋 경로로 설정
    
    * Evaluation (Validation case)만 진행하는 경우
        
        * `bash inference.sh --qa ext --retrieval sparse --do_eval`

    * Predictions를 저장하고 싶은 경우

        * `bash inference.sh --qa ext --retrieval sparse --do_predict`

        * `bash inference.sh --qa ext --retrieval sparse --do_predict --do_valid` (Validation 결과도 같이 저장)


## Trouble Shooting

### Import error

inference를 하는 과정과 Retrieval를 학습하는 과정에서 발생

* `utils.utils_sparse_retrieval` 모듈을 찾을 수 없는 현상

    * 현재 경로문제 해결 진행 중... 임시방편으로 아래 방법 활용

    * `/retrieval/sparse/sparse_retrieval.py` 에서 import 부분에 `utils.utils_sparse_retrieval`를 `utils_sparse_retrieval`로만 import

        * 반대로 위 모듈을 찾을 수 없다면, `utils.`을 붙여서 import

    > `utils.`을 붙이거나 떼거나 두 경우를 테스트해서 실행되는 코드로 진행
