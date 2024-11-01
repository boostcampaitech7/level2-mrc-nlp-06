import logging
import os
import sys
import random
import numpy as np
import torch
import argparse

sys.path.append("/data/ephemeral/home/jh/level2-mrc-nlp-06/src")

from typing import NoReturn
from datasets import DatasetDict, load_from_disk, load_metric
from trainer_mrc import QuestionAnsweringTrainer
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorWithPadding,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
import nltk

from utils.utils_mrc import (
    check_no_error,
    postprocess_qa_predictions,
    json_to_Arguments,
)

seed = 2024
deterministic = False

random.seed(seed)  # python random seed 고정
np.random.seed(seed)  # numpy random seed 고정
torch.manual_seed(seed)  # torch random seed 고정
torch.cuda.manual_seed_all(seed)
if deterministic:  # cudnn random seed 고정 - 고정 시 학습 속도가 느려질 수 있습니다.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

logger = logging.getLogger(__name__)


def main():
    # Load Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        help="config 폴더에 있는 json 파일을 로드하세요",
        default="../../config/base_config.json",
    )
    parser.add_argument(
        "--do_train",
        action="store_true",
        required=True,
        help="train을 할 경우에 활성화",
    )
    parser.add_argument(
        "--do_eval", action="store_true", help="Eval도 같이 할 경우에 활성화"
    )
    args = parser.parse_args()
    model_args, data_args, training_args = json_to_Arguments(args.config)

    training_args = TrainingArguments(**training_args)

    if args.do_train:
        training_args.do_train = True
    else:
        training_args.do_train = False

    if args.do_eval:
        training_args.do_eval = True
    else:
        training_args.do_eval = False

    print(f"model is from {model_args.model_name_or_path}")
    print(f"data is from {data_args.train_dataset_name}")

    set_seed(training_args.seed)

    datasets = load_from_disk(data_args.train_dataset_name)
    print(datasets)

    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        # 'use_fast' argument를 True로 설정할 경우 rust로 구현된 tokenizer를 사용할 수 있습니다.
        # False로 설정할 경우 python으로 구현된 tokenizer를 사용할 수 있으며,
        # rust version이 비교적 속도가 빠릅니다.
        use_fast=True,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
    )

    print(
        type(training_args),
        type(model_args),
        type(datasets),
        type(tokenizer),
        type(model),
    )
    print("***** Train Abstractive QA *****")
    training_args.output_dir = str(
        os.path.join(training_args.output_dir, "Abstractive")
    )
    run_mrc_generation(data_args, training_args, model_args, datasets, tokenizer, model)


def run_mrc_generation(
    data_args: dict,
    training_args: TrainingArguments,
    model_args: dict,
    datasets: DatasetDict,
    tokenizer,
    model,
) -> NoReturn:

    if training_args.do_train:
        column_names = datasets["train"].column_names
    else:
        column_names = datasets["validation"].column_names

    def preprocess_function(examples):
        inputs = [
            f"question: {q}  context: {c} </s>"
            for q, c in zip(examples["question"], examples["context"])
        ]
        targets = [f'{a["text"][0]} </s>' for a in examples["answers"]]
        model_inputs = tokenizer(
            inputs,
            max_length=data_args.max_source_length,
            padding="max_length",
            truncation=True,
        )

        ###TODO: targets(label)을 위해 tokenizer 설정###
        labels = tokenizer(
            targets,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        model_inputs["labels"] = labels["input_ids"]
        model_inputs["example_id"] = []
        for i in range(len(model_inputs["labels"])):
            model_inputs["example_id"].append(examples["id"][i])
        return model_inputs

    last_checkpoint, max_seq_length = check_no_error(
        data_args, training_args, datasets, tokenizer
    )

    column_names = datasets["train"].column_names
    train_dataset = datasets["train"]
    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=False,
    )

    if training_args.do_eval:
        eval_dataset = datasets["validation"]
        eval_dataset = eval_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=False,
        )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
    )

    def post_process_function(examples, predictions):
        """
        examples: 원본
        features: 전처리된 데이터셋
        predictions: 모델 예측 결과
        """
        # 1. 예측한 prediction token Decoding
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        # label은 따로 구하지 않음 (Rouge metric 용도)
        # labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        # decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # 2. 공백 제거, 문장 단위 분리
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in decoded_preds]

        # 3. Format 맞춰주기
        formatted_predictions = [
            {"id": ex["id"], "prediction_text": decoded_preds[i]}
            for i, ex in enumerate(examples)
        ]
        references = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]

        # 4. 후처리 결과 작업 저장

        return formatted_predictions, references

    metric = load_metric("squad")

    def compute_metrics(eval_preds):
        predictions, labels = eval_preds
        formatted_predictions, references = post_process_function(
            datasets["validation"], predictions
        )
        result = metric.compute(
            predictions=formatted_predictions, references=references
        )
        return result

    args = Seq2SeqTrainingArguments(
        output_dir=training_args.output_dir,
        do_train=training_args.do_train,
        do_eval=training_args.do_eval,
        predict_with_generate=True,
        per_device_train_batch_size=training_args.train_batch_size,
        per_device_eval_batch_size=training_args.eval_batch_size,
        num_train_epochs=training_args.num_train_epochs,
        save_strategy="epoch",
        save_total_limit=1,  # 모델 checkpoint를 최대 몇개 저장할지 설정
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Training
    if training_args.do_train:
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif os.path.isdir(model_args.model_name_or_path):
            checkpoint = model_args.model_name_or_path
        else:
            checkpoint = None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")

        with open(output_train_file, "w") as writer:
            logger.info("***** Train results *****")
            for key, value in sorted(train_result.metrics.items()):
                logger.info(f"  {key} = {value}")
                writer.write(f"{key} = {value}\n")

        # State 저장
        trainer.state.save_to_json(
            os.path.join(training_args.output_dir, "trainer_state.json")
        )

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(max_length=128, num_beams=3)

        metrics["eval_samples"] = len(eval_dataset)

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
