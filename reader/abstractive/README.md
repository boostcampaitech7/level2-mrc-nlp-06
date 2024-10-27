## Abstractive (Generative) Question Answering
- 생성형 언어 모델을 활용하여 검색된 문서로부터 질문에 대한 답을 생성하는 코드입니다.
- T5(`train_abs.py`), Llama 3.1(`train_llama.py`), OpenAI GPT 4 모델(``)을 각각 활용하여 답변을 생성할 수 있습니다.
- **T5**
    - denoising autoencoder 모델인 T5 모델을 fine-tuning하여 extractive QA에 적절한 형태로 답을 생성합니다.
- **Llama 3.1 with Q-LoRA**
    - [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) 모델을 Q-LoRA 방식으로 fine-tuning합니다.
    - 이를 통해 질문의 키워드가 아닌, 전체적인 맥락을 고려하며 답변을 생성할 것으로 기대합니다.
    - 효율적인 양자화(Q-LoRA) 기법을 사용하여 높은 정확도와 낮은 메모리 사용량을 실현합니다.
    - 이후 ODQA inference에서 모델 Fine-tuning과 inference & evaluation이 구분되어 수행되는 점을 반영하여 `train_llama.py`에서도 두 과정을 분리했습니다.
- **OpenAI GPT 4 API**
    - OpenAI API를 통해 GPT-4 모델을 활용하여 질문에 대한 답변을 생성합니다.
    - API 호출 방식으로 편리하게 다양한 데이터셋에 접근하여 QA 작업을 수행합니다.


## 폴더 구조
```text
.
├── README.md
├── config
│   ├── base_gen_config.json
│   └── llama_config.json
├── prompt
│   ├── eng_v1.txt
│   ├── eng_v2.txt
│   └── kor_v1.txt
└── src
    ├── __init__.py
    ├── train_abs.py
    └── train_llama.py
```


## How To
### `train_llama.py` 사용법
`./reader/abstractive` 디렉토리에서 다음 명령어를 사용해 실행합니다.
```bash
python src/train_llama.py --config {config_file_name}
```

### config 예시
[config](./config/) 폴더에 각 모델별 config 예시 파일이 저장되어 있습니다. 구체적인 내용은 다음과 같습니다.
- [llama_config.json](./config/llama_config.json) 설정 파일 주요 항목:
    - `model_name`: 사용할 Llama 모델 (기본값: `"meta-llama/Llama-3.1-8B-Instruct"`)
    - `data_path`: 학습 및 평가 데이터 경로
    - `prompt_path`: Llama 모델에 사용할 QA 프롬프트 경로
    - `output_dir`: 학습 완료된 모델과 validation 데이터 예측 결과 (`predictions.json`) 저장 경로
    - `bnb_4bit_quant_type`: 양자화 유형 (기본값: `nf4`)
    - `num_train_epochs`: 학습 epoch 수
    - `bert_model`: BertScore 계산을 위해 사용하는 모델 (기본값: `distilbert-base-uncased`)

### outputs
- fine-tuned `llama` 모델의 adapter layer 가중치와 validation 데이터에 대한 predictions 결과는 config 파일에서 지정한 `output_dir`에 저장됩니다.


## Reference
- [Fine Tuning Llama 3.2 11B for Question Answering](https://medium.com/@coldstart_coder/fine-tuning-llama-3-2-11b-for-question-answering-435c28bb57c1)
- [Hugging Face Llama Recipes](https://github.com/huggingface/huggingface-llama-recipes?tab=readme-ov-file)