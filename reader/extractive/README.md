## Extractive Model

- Reader 모델 중 Extractive 방법으로 Start point, End point의 Token들을 예측하는 방법

### 모델 설명

- `uomnf97/klue-roberta-finetuned-korquad-v2`
  - RoBERTa 모델에서 KorQuad v2 기반으로 Fintuned된 모델
- `monologg/koelectra-base-v3-finetuned-korquad`
  - KoElectra-base 모델에서 KorQuad v1 기반으로 Fintuned된 모델

## 폴더 구조

```text
extractive
|-- README.md
|-- config
|   |-- base_config.json
|   `-- koelectra.json
|-- model (train 후)
|   |-- Extractive
|-- src
|   |-- train_ext.py
|   `-- trainer_ext.py
`-- utils
	|-- preprocessing.py
    `-- utils_mrc.py
```



## How To

### Config  설정

[config](https://github.com/boostcampaitech7/level2-mrc-nlp-06/blob/develop/reader/extractive/config/base_config.json)는 총 3가지로 config가 이루어져있다.

1. `model_args` : 모델이 저장된 폴더 or huggingface 에 등록된 모델명을 불러오는 args
2. `data_args` : Train, Valid, Test 데이터셋이 있는 path와 preprocessing할 때 필요한 값들을 불러오는 args
   * Data processing에 필요한 Hyperparameter를 설정
3. `training_args` : transformers의 Trainer에 필요한 `TrainingArguments` 인자들을 설정해주는 args
   * Train에 필요한 Hyperparameter를 설정

### 실행

최상위 폴더에서의 [train_mrc.sh](https://github.com/boostcampaitech7/level2-mrc-nlp-06/blob/develop/train_mrc.sh) 실행

* `train_mrc.sh` 코드 내에 위에서 생성한 config 파일의 path 설정

  ```shell
  config_name="./reader/extractive/config/base_config.json"
  ```

* bash 파일 실행

    ```shell
    level2-mrc-nlp-06$ bash train_mrc.sh --type=ext --do_eval
    ```

    * `--do_eval` : 평가 metrics를 출력할지에 대한 여부 설정

### 결과 저장

- 학습된 모델은 `./{training_args.output_dir}/Extractive`에 저장됩니다.

* 추후 inference 과정에서 해당 경로가 사용됨



## Trouble Shooting

### BERT계열 모델 사용

현재 `utils/preprocessing.py` 는 RoBERTa 계열의 모델로 설정되어 있음

BERT계열의 모델 사용시 `return_token_type_ids` 옵션을 `TRUE`로 설정해주어야함

```python
# preprocessing.py
def prepare_train_features(examples): # prepare_valid_features도 동일하게 설정
    tokenized_examples = tokenizer(
        ...
        return_token_type_ids=true, # roberta모델을 사용할 경우 False, bert를 사용할 경우 True로 표기해야합니다.
		...
    )
```
