# # ======= nested ======= #
import logging
import sys
import os
sys.path.append('/data/ephemeral/home/jh/level2-mrc-nlp-06/ODQA')
sys.path.append('/data/ephemeral/home/jh/level2-mrc-nlp-06/reader')

# ======= data processing ======= #
import numpy as np
import pandas as pd
from typing import Callable, Dict, List, NoReturn, Tuple

# ======= extractive model ======= #
from reader.utils.utils_mrc import json_to_Arguments, postprocess_qa_predictions
from reader.extractive.src.trainer_ext import QuestionAnsweringTrainer


# ======= Hugging face ======= #
from datasets import (
    Dataset,
    DatasetDict,
    Features,
    Sequence,
    Value,
    load_from_disk,
    load_metric,
)
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    TrainingArguments,
    set_seed,
)


class ExtractiveQA():
    
    def __init__(self, args, config_path, test_datasets, valid_datasets):
        self.args = args
        self.config_path = config_path
        self.test_datasets = test_datasets
        self.valid_datasets = valid_datasets

        self.model = None
        self.tokenizer = None
        self.model_config = None

        config_args = self.load_model()
        
        self.model_args = config_args[0]
        self.data_args = config_args[1]
        self.training_args = config_args[2]

    
        self.logger = logging.getLogger(__name__)

    def load_model(self):
        
        model_args, data_args, training_args = json_to_Arguments(self.config_path)

        # Model 불러오기
        self.model_config = AutoConfig.from_pretrained(
            model_args.config_name
            if model_args.config_name
            else model_args.model_name_or_path,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name
            if model_args.tokenizer_name
            else model_args.model_name_or_path,
            use_fast=True,
        )
        self.model = AutoModelForQuestionAnswering.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=self.model_config,
        )
        
        return model_args, data_args, training_args
    
    def predict(self, datasets, output_dir):
        # QA 모델의 경우 Retrieval의 결과가 필요함
        # Datasets : Retrieval 결과에 대한 datasets

        # model inference 결과
        # eval 혹은 prediction에서만 사용함
        column_names = datasets["validation"].column_names

        question_column_name = "question" if "question" in column_names else column_names[0]
        context_column_name = "context" if "context" in column_names else column_names[1]
        answer_column_name = "answers" if "answers" in column_names else column_names[2]

        # Padding에 대한 옵션을 설정합니다.
        # (question|context) 혹은 (context|question)로 세팅 가능합니다.
        pad_on_right = self.tokenizer.padding_side == "right"

        # Validation preprocessing / 전처리를 진행합니다.
        def prepare_validation_features(examples):
            # truncation과 padding(length가 짧을때만)을 통해 toknization을 진행하며, stride를 이용하여 overflow를 유지합니다.
            # 각 example들은 이전의 context와 조금씩 겹치게됩니다.
            tokenized_examples = self.tokenizer(
                examples[question_column_name if pad_on_right else context_column_name],
                examples[context_column_name if pad_on_right else question_column_name],
                truncation="only_second" if pad_on_right else "only_first",
                max_length=self.model_args.max_seq_length,
                stride=self.data_args.doc_stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                return_token_type_ids=False, # roberta모델을 사용할 경우 False, bert를 사용할 경우 True로 표기해야합니다.
                padding="max_length" if self.data_args.pad_to_max_length else False,
            )

            # 길이가 긴 context가 등장할 경우 truncate를 진행해야하므로, 해당 데이터셋을 찾을 수 있도록 mapping 가능한 값이 필요합니다.
            sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

            # evaluation을 위해, prediction을 context의 substring으로 변환해야합니다.
            # corresponding example_id를 유지하고 offset mappings을 저장해야합니다.
            tokenized_examples["example_id"] = []

            for i in range(len(tokenized_examples["input_ids"])):
                # sequence id를 설정합니다 (to know what is the context and what is the question).
                sequence_ids = tokenized_examples.sequence_ids(i)
                context_index = 1 if pad_on_right else 0

                # 하나의 example이 여러개의 span을 가질 수 있습니다.
                sample_index = sample_mapping[i]
                tokenized_examples["example_id"].append(examples["id"][sample_index])

                # context의 일부가 아닌 offset_mapping을 None으로 설정하여 토큰 위치가 컨텍스트의 일부인지 여부를 쉽게 판별할 수 있습니다.
                tokenized_examples["offset_mapping"][i] = [
                    (o if sequence_ids[k] == context_index else None)
                    for k, o in enumerate(tokenized_examples["offset_mapping"][i])
                ]
            return tokenized_examples
        
        eval_dataset = datasets["validation"] # 수정 필요

        # Validation Feature 생성
        eval_dataset = eval_dataset.map(
            prepare_validation_features,
            batched=True,
            num_proc=self.data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not self.data_args.overwrite_cache,
        )
        # Data collator
        # flag가 True이면 이미 max length로 padding된 상태입니다.
        # 그렇지 않다면 data collator에서 padding을 진행해야합니다.
        data_collator = DataCollatorWithPadding(
            self.tokenizer, pad_to_multiple_of=8 if self.training_args.fp16 else None
        )

        # Post-processing:
        def post_processing_function(
            examples,
            features,
            predictions: Tuple[np.ndarray, np.ndarray],
            training_args: TrainingArguments,
        ) -> EvalPrediction:
            # Post-processing: start logits과 end logits을 original context의 정답과 match시킵니다.
            predictions = postprocess_qa_predictions(
                examples=examples,
                features=features,
                predictions=predictions,
                max_answer_length=self.data_args.max_answer_length,
                output_dir=training_args.output_dir,
            )
            # Metric을 구할 수 있도록 Format을 맞춰줍니다.
            formatted_predictions = [
                {"id": k, "prediction_text": v} for k, v in predictions.items()
            ]

            if training_args.do_predict:
                return formatted_predictions
            elif training_args.do_eval:
                references = [
                    {"id": ex["id"], "answers": ex[answer_column_name]}
                    for ex in datasets["validation"]
                ]

                return EvalPrediction(
                    predictions=formatted_predictions, label_ids=references
                )

        metric = load_metric("squad")

        def compute_metrics(p: EvalPrediction) -> Dict:
            return metric.compute(predictions=p.predictions, references=p.label_ids)

        # Trainer 초기화
        trainer = QuestionAnsweringTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=None,
            eval_dataset=eval_dataset,
            eval_examples=datasets["validation"],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            post_process_function=post_processing_function,
            compute_metrics=compute_metrics
        )

        self.logger.info("*** Evaluate ***")

        #### eval dataset & eval example - predictions.json 생성됨
        if self.training_args.do_predict:
            predictions = trainer.predict(
                test_dataset=eval_dataset, test_examples=datasets["validation"]
            )

            # predictions.json 은 postprocess_qa_predictions() 호출시 이미 저장됩니다.
            print(
                "No metric can be presented because there is no correct answer given. Job done!"
            )

        if self.training_args.do_eval:
            metrics = trainer.evaluate()
            metrics["eval_samples"] = len(eval_dataset)

            trainer.log_metrics("test", metrics)
            trainer.save_metrics("test", metrics)
        
        return predictions