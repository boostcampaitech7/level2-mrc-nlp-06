# ODQA Inference

ODQA Task를 Inference하는 코드

* Retrieval에서 top-k개 만큼 선정한 Context들을 기반으로 Reader에서 Start Point, End Point의 Token들을 예측하는 시스템
* `predicionts.json`과 `n-best_predictions.json`을 저장



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

* Retrieval에서

* [REAMDE.md 참고](https://github.com/boostcampaitech7/level2-mrc-nlp-06/tree/develop/reader/extractive/README.md)


### 3. Inference 실행

* 최상위 폴더에서 `inference.sh` 실행


    * `ODQA/config` 파일 설정

        * `qa_config`는 reader 모델의 config 위치로 설정

        * `retrieval_config`는 사용했던 config 파일 이름으로 설정

        * `datasets` 는 데이터셋 경로로 설정

    * Evaluation (Validation case)만 진행하는 경우

      * ```shell
      $ bash inference.sh --qa ext --retrieval sparse --do_eval

      Predictions를 저장하고 싶은 경우

      * ```shell
        $ bash inference.sh --qa ext --retrieval sparse --do_predict
        ```
        * `ODQA/predictions/predict`에 `predictions.json, n-best predictions.json` 저장
      * ```shell
        $ bash inference.sh --qa ext --retrieval sparse --do_predict --do_valid # (Validation 결과도 같이 저장)
        ```

	    * `ODQA/predictions/validation`에도 `predictions.json, n-best predictions.json, predictions` 저장
        * validation의 경우 실제 정답 (`answer`)도 포함되어있음



## Ensemble

Ensemble은 Hard Voting 방식으로 구현됨

### Hard Voting

`ODQA/predictions/ensemble/pred` 에 저장되어있는 각 `predictions.json` 에서 `id`마다 예측한 `ans` 들을 취합하여 가장 많이 예측된 단어를 선택하는 방식으로 구현

```python
def get_predictions(path):

    predictions_per_id = {}
    print(" ***** Predictions for Ensemble ***** ")
    # predictions_{ver}.json들 가져오기
    for names in os.listdir(save_dir):
        data_path = os.path.join(save_dir, names)
        print(names)
        with open(data_path,"r") as f:
            pred_data = json.load(f)
        for id in pred_data:
            if id in predictions_per_id:
                predictions_per_id[id].append(pred_data[id])
            else:
                predictions_per_id[id] = [pred_data[id]]

    return predictions_per_id

def get_preds_json(preds, result_dir):
    ensemble = {}
    for id, answers in preds.items():
        counts = Counter(answers)
        ensemble[id] = counts.most_common(1)[0][0]
        print(counts)
    with open(os.path.join(result_dir,"ensemble.json"),"w") as f:
        json.dump(ensemble, f, ensure_ascii=False, indent=4)

    print(" ***** ensemble task complete ***** ")
```

* `get_predictions()` : json 파일들에서 `id`별 정답을 추출
* `get_preds_json()` : `Counter` 함수를 통해 예측된 단어들의 갯수들을 구하고 가장 많이 나온 단어를 선택



## Module 설명

### `run_odqa()`

```python
# inference.py

def run_odqa(args, inference_config, test_datasets, wiki_path, pred_dir, valid_datasets=None):
    # Load QA Model
    match args.qa:
        case "ext":
            print("******* Extractive QA Model 선택 *******")
            QA_model = ExtractiveQA(args, inference_config.qa_config, test_datasets, valid_datasets)
        case "abs":
            pass

    # Load Retrieval Model
    match args.retrieval:
        case "sparse":
            print("******* Sparse Retrieval Model 선택 *******")
            Ret_model = Sparse_Model(args, inference_config, test_datasets, wiki_path, valid_datasets)
        case "dense":
            print(("******* Dense (SBERT) Retrieval Model 선택 *******"))
            Ret_model = Dense_Model(args, inference_config, test_datasets, wiki_path, valid_datasets)
            pass
        case "hybrid":
            pass
```

* `--qa` 와 `--retrieval` 을 설정한 타입에 따라 각 모델들을 호출
  * 현재 abs(generative) 타입과 hybrid 타입에 대한 코드는 미구현
* 각 모델들은 `modules/` 에 구현되어있음
  * `extractiveQA.py, sparseRetrieval, denseRetrieval.py`

```python
if args.do_predict:
    # Inference
    output_dir = os.path.join(pred_dir, "test") # predictions.json 출력폴더

    # 1. Retrieval Model에서 Question에 맞는 Context를 가져옴
    test_retrieve_datasets = Ret_model.get_contexts(test_datasets)

    # 2. Reader에 Context를 전달하여 Inference를 수행
    # Extraction Reader의 경우 postprocess_qa_predictions함수를 수행하면서 자동적으로 Json 저장
    test_predictions = QA_model.predict(test_retrieve_datasets, output_dir)

    # 3. 만약에 Validation도 추가로 Predictions을 뽑고 싶을 때
    if args.do_valid:
        output_dir = os.path.join(pred_dir, "validation")
        valid_retrieve_datasets = Ret_model.get_contexts(valid_datasets)
        valid_predictions = QA_model.predict(valid_retrieve_datasets, output_dir)
        dict_to_json(os.path.join(output_dir,"predictions.json"),
                     os.path.join(output_dir,"prediction_with_ans.json"),
                     answers)

elif args.do_eval:
    output_dir = os.path.join(pred_dir, "eval")
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

        valid_retrieve_datasets = Ret_model.get_contexts(valid_datasets)
        QA_model.predict(valid_retrieve_datasets, output_dir)
```

* `Ret_model.get_contexts()` 에서 retrieval된 결과(Dataframe)를 `Datasets.Features` 로 변경
* `QA_model.predict()` 에서는 `Ret_model` 에서 나온 contexts들을 기반으로 단어를 예측함
* `--do_predict` 옵션을 선택시 `reader`에서 추출한 값을 기반으로 `predictions.json`을 생성하는 코드
  * 저장하는 코드는 `QA_model`이 predict하면서 후처리하는 함수에서 생성
  * `--do_valid` 도 마찬가지로, validation dataset에 대해 predictions를 수행
    * 정답과 함께 포함하는 경우에 대해 따로 처리를 해주는 `dict_to_json` 함수 실행

> `--do_eval` 과 `--do_predict`는 같이 사용할 수 없음





## Trouble Shooting

### Import error

inference를 하는 과정과 Retrieval를 학습하는 과정에서 발생

* `utils.utils_sparse_retrieval` 모듈을 찾을 수 없는 현상

    * 현재 경로문제 해결 진행 중... 임시방편으로 아래 방법 활용

    * `/retrieval/sparse/sparse_retrieval.py` 에서 import 부분에 `utils.utils_sparse_retrieval`를 `utils_sparse_retrieval`로만 import
    * 반대로 위 모듈을 찾을 수 없다면, `utils.`을 붙여서 import

    ```python
    # /retrieval/sparse/sparse_retrieval.py

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils_sparse_retrieval import timer, hit, mrr # opt.1
    from utils.utils_sparse_retrieval import timer, hit, mrr # opt.2
    ```

    > `utils.`을 붙이거나 떼거나 두 경우를 테스트해서 실행되는 코드로 진행
