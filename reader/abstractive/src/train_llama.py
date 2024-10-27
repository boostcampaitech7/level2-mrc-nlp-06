import sys
import logging
import json
from box import Box
import argparse
from datasets import load_from_disk, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, pipeline
from peft import prepare_model_for_kbit_training, LoraConfig
from trl import SFTTrainer
from evaluate import load
import torch
import gc


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -    %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# 주어진 경로에서 JSON 형식의 설정 파일을 로드하여 반환
def load_config(config_path: str):
    with open(config_path, "r") as f:
        config = json.load(f)
        config = Box(config)
    return config

# Llama conversation format으로 데이터 변환 
def convert_squad_sample_to_llama_conversation(sample, tokenizer, instruction_prompt_template):
    question = sample['question']
    context = sample['context']
    answers = sample['answers']['text']
    if len(answers) == 0 :
      answer = "The context does not provide an answer..."
    else:
      answer = sample['answers']['text'][0]

    messages = [
        {"role": "system", "content": instruction_prompt_template.format(context=context)},
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer}
    ]
    sample_conversation = tokenizer.apply_chat_template(messages, tokenize=False)
    return {"text": sample_conversation, "messages": messages, "answer": answer}

# batch 단위로 inference
def get_base_and_tuned_bulk_predictions(pipe, samples): 
    bulk_messages = [i[:-1] for i in samples['messages']]
    responses = pipe(bulk_messages, max_new_tokens=64, batch_size=len(samples), do_sample=False)
    responses = [i[0]['generated_text'][-1]['content'] for i in responses]
    return {"predicted_answer": responses}



def main():
    # Training --------------------------------------------------------------

    # argument 로딩
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, \
                        help="config 폴더에 있는 json 파일을 로드하세요", \
                        default="/data/ephemeral/home/minji/reader/config/llama_config.json")
    config = parser.parse_args()
    args = load_config(config.config)
    logger.info(f"model is from {args.model_name}")  # llama 모델명 출력
    logger.info(f"data is from {args.data_path}")    # 데이터 경로 출력

    # 데이터셋 불러오기
    datasets = load_from_disk(args.data_path)
    train_dataset = datasets['train']
    eval_dataset = datasets['validation']

    # # korquad v1.0 데이터를 추가로 이용해 fine-tuning
    # korquad_datasets = load_from_disk("/data/ephemeral/home/datasets/korquad") 

    # # korquad train과 validation을 절반씩 추출 (시간관계 상)
    # korquad_train_dataset_split = korquad_datasets['train'].train_test_split(train_size=0.5)
    # korquad_eval_dataset_split = korquad_datasets['validation'].train_test_split(train_size=0.5)

    # # train과 eval의 train 부분만 결합
    # korquad_train_dataset = korquad_train_dataset_split['train']
    # korquad_eval_dataset = korquad_eval_dataset_split['train']

    # # 기존 데이터셋과 korquad 데이터셋 결합
    # train_dataset = datasets['train'].remove_columns('document_id')
    # eval_dataset = datasets['validation'].remove_columns('document_id')
    # train_dataset = concatenate_datasets([train_dataset, korquad_train_dataset, korquad_eval_dataset])

    logger.info(f"{len(train_dataset)} data are used for training.")  # 데이터셋 크기 출력
    
    # 프롬프트 불러오기
    with open(args.prompt_path, "r") as f:
        instruction_prompt_template = f.read()
    
    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    # Llama 학습을 위한 형태로 데이터 변환
    conversation_training_samples = train_dataset.map(lambda sample: convert_squad_sample_to_llama_conversation(sample, tokenizer, instruction_prompt_template))
    conversation_validation_samples = eval_dataset.map(lambda sample: convert_squad_sample_to_llama_conversation(sample, tokenizer, instruction_prompt_template))

    # Q-LoRA를 위한 config 지정
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config = bnb_config,
        device_map="auto"
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False   # 학습 중이고, output이 변경될 것으로 예상되므로 False

    # PEFT config 정의
    rank = 128
    alpha = rank*2
    peft_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.05, # dropout for the lora layers while training, to avoid overfitting
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=['k_proj', 'q_proj', 'v_proj', 'o_proj', 'gate_proj', 'down_proj', 'up_proj']
    )

    # Training Argument 정의
    training_arguments = TrainingArguments(
        output_dir=args.output_dir+"/checkpoints",
        optim='paged_adamw_32bit',
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        log_level='debug',
        evaluation_strategy = "epoch",
        save_strategy='epoch',
        logging_steps=8,
        eval_steps=8,
        learning_rate=1e-4,
        fp16=True,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=0.1,
        load_best_model_at_end = True,
        overwrite_output_dir = True,
        lr_scheduler_type='linear',
    )

    # Supervised Fine-tuning을 위한 Trainer 정의
    trainer = SFTTrainer(
        model=model,
        train_dataset=conversation_training_samples,
        eval_dataset=conversation_validation_samples,
        peft_config=peft_config,
        dataset_text_field='text', # 학습 시 참조할 text 값에 대한 key
        max_seq_length=512, # 최대 몇 개의 토큰까지 생성할건지
        tokenizer=tokenizer,
        args=training_arguments
    )
    trainer.model.print_trainable_parameters()

    # fine-tuning
    trainer.train()
    trainer.save_model(args.output_dir)  # 학습된 weight 저장: adapter weight만 학습했으므로 이 adapter만 저장됨
    
    # 불필요한 변수 삭제
    model.cpu()
    del model, tokenizer, trainer
    gc.collect()
    torch.cuda.empty_cache()


    # Evaluation --------------------------------------------------------------

    # 토크나이저 로드 
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    # Q-LoRA를 위한 config 지정
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config = bnb_config,
        device_map="auto"
    )

    # 학습된 adapter를 pre-trained llama 모델에 추가
    model.load_adapter(args.output_dir, adapter_name="adapter")
    model.enable_adapters() 

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    # text-generation을 위해 huggingface pipeline으로 모델 감싸기
    model_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    # validation 데이터셋에 대해 inference
    conversation_validation_samples = conversation_validation_samples.map(lambda samples: get_base_and_tuned_bulk_predictions(pipe=model_pipe, samples=samples), batched=True, batch_size=5)

    # prediction 결과 저장
    result_dict = {}  # id별로 answer와 predicted_answer 저장할 계층형 딕셔너리 생성
    for sample in conversation_validation_samples:
        sample_id = sample['id']
        answer = sample['answer']
        predicted_answer = sample['predicted_answer']
        
        result_dict[sample_id] = {
            'answer': answer,
            'predicted_answer': predicted_answer
        }

    output_file = args.output_dir + "/predictions.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=4)

    logger.info(f"Data saved to {output_file}")


    # BERT Score 계산
    bertscore = load("bertscore")
    bert_predictions = conversation_validation_samples['predicted_answer']
    bert_references = conversation_validation_samples['answer']
    trained_validation_bert_score = bertscore.compute(predictions=bert_predictions, references=bert_references, lang="kr", model_type=args.bert_model, device="cuda:0")
    tuned_averages = {key: sum(trained_validation_bert_score[key])/len(trained_validation_bert_score[key]) for key in ['precision', 'recall', 'f1']}

    # EM, F1-score 계산
    em_f1_score = load("squad")
    ex_predictions = [{'id': sample['id'], 'prediction_text': sample['predicted_answer']} for sample in conversation_validation_samples]
    ex_references = [{'id': sample['id'], 'answers': [{'text': sample['answers']['text'][0], 'answer_start': sample['answers']['answer_start'][0]}]} for sample in conversation_validation_samples]
    tuned_exact_match_score = em_f1_score.compute(predictions=ex_predictions, references=ex_references)
    tuned_averages['exact_match'] = tuned_exact_match_score['exact_match']
    tuned_averages['squad_f1'] = tuned_exact_match_score['f1']

    logger.info(tuned_averages)


if __name__ == "__main__":
	main()