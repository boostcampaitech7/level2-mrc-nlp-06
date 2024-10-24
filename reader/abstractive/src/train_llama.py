# reference
# 1. https://medium.com/@coldstart_coder/fine-tuning-llama-3-2-11b-for-question-answering-435c28bb57c1
# 2. https://github.com/huggingface/huggingface-llama-recipes?tab=readme-ov-file

# metric
# - bert-score: 모델 출력과 예상 답변 사이의 의미적 유사성 측정
# - EM, F1-score: 정확한 답변 반환 여부

# Model and Dataset Access: huggingface-cli login

# Data Preparation ------------------------------------------

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

def get_base_and_tuned_bulk_predictions(pipe, samples): 
    bulk_messages = [i[:-1] for i in samples['messages']]
    responses = pipe(bulk_messages, max_new_tokens=64, batch_size=len(samples), do_sample=False)
    responses = [i[0]['generated_text'][-1]['content'] for i in responses]
    return {"predicted_answer": responses}



def main():
    
    # 기본 세팅 -----------------------------------------------------------------

    # Load Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, \
                        help="config 폴더에 있는 json 파일을 로드하세요", \
                        default="/data/ephemeral/home/minji/reader/config/llama_config.json")
    config = parser.parse_args()
    args = load_config(config.config)

    # 모델, 데이터 정보 출력
    logger.info(f"model is from {args.model_name}")
    logger.info(f"data is from {args.data_path}")

    # 데이터셋 불러오기
    datasets = load_from_disk(args.data_path)
    korquad_datasets = load_from_disk("/data/ephemeral/home/datasets/korquad")

    # 기존 train과 eval 데이터셋에서 'document_id' 열 제거
    train_dataset = datasets['train'].remove_columns('document_id')
    eval_dataset = datasets['validation'].remove_columns('document_id')

    # korquad train과 validation을 절반씩 추출 (시간관계 상)
    korquad_train_dataset_split = korquad_datasets['train'].train_test_split(train_size=0.5)
    korquad_eval_dataset_split = korquad_datasets['validation'].train_test_split(train_size=0.5)

    # train과 eval의 train 부분만 결합
    korquad_train_dataset = korquad_train_dataset_split['train']
    korquad_eval_dataset = korquad_eval_dataset_split['train']

    # 기존 데이터셋과 korquad 데이터셋 결합
    train_dataset = concatenate_datasets([train_dataset, korquad_train_dataset, korquad_eval_dataset])

    # 데이터셋 크기 출력
    logger.info(f"{len(train_dataset)} data are used for training.")
    

    with open(args.prompt_path, "r") as f:
        instruction_prompt_template = f.read()
    
    # 모델 및 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    # Llama 학습을 위한 형태로 데이터 변환
    conversation_training_samples = train_dataset.map(lambda sample: convert_squad_sample_to_llama_conversation(sample, tokenizer, instruction_prompt_template))
    conversation_validation_samples = eval_dataset.map(lambda sample: convert_squad_sample_to_llama_conversation(sample, tokenizer, instruction_prompt_template))


    # Model Preparation Preparation ------------------------------------------

    # load the base model with 4-bit precision to help save on the gpu overhead.
    # setup our config for the LoRA weights
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,  ###TODO: 다른 옵션 테스트 
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
    model.config.use_cache = False   # 학습 중이고, output이 변경될 것으로 예상되므로 

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

    # Supervised Training -----------------------------------------------------

    # Loss function은 여기에서 정의되지 않고, 허깅페이스 모델에 대한 파라미터로서 저장되어 있음
    # Llama의 경우 loss function은 cross entropy loss로 적용됨

    # first define some training arguments
    training_arguments = TrainingArguments(
        output_dir=args.output_dir+"/checkpoints",
        optim='paged_adamw_32bit', #specify what optimizer we wwant to use, in this case a 8bit version of adamw with pagination.
        per_device_train_batch_size=8, # define the number of samples per training batch
        gradient_accumulation_steps=4, # define how many steps to accumulate gradients,
        log_level='debug',
        evaluation_strategy = "epoch",
        save_strategy='epoch', # we'll save a checkpoint every epoch
        logging_steps=8,
        eval_steps=8,
        learning_rate=1e-4, # for llm training we want a fairly high learning rate, 1e-4 is a good starting point but it's worth it to play around with this value
        fp16=True,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=0.1,
        load_best_model_at_end = True,
        overwrite_output_dir = True,
        lr_scheduler_type='linear',# and set our learning rate decay
    )

    # now that we have our arguments, we'll use that to create our trainer,
    # passing in the model, dataset, peft config, tokenizer, etc
    trainer = SFTTrainer(
        model=model,
        train_dataset=conversation_training_samples,
        eval_dataset=conversation_validation_samples,
        peft_config=peft_config,
        dataset_text_field='text', # datasets always has samples in a dictionary, so we need to specify what key to reference when training
        max_seq_length=512, # specify how many tokens to generate per training, this is just so it doesn't generate for forever especially for shorter samples
        tokenizer=tokenizer,
        args=training_arguments
    )
    trainer.model.print_trainable_parameters()

    # 학습 및 학습된 weight 저장
    trainer.train()
    trainer.save_model(args.output_dir)  # adapter weight만 학습했으므로 이 adapter만 저장됨
    
    # move the model to the cpu and then delete the model, tokenizer and trainer objects
    model.cpu()
    del model, tokenizer, trainer
    # We'll also call python to garbage collect any resources that might
    # still be hanging around, and we'll clear the cuda cache.
    gc.collect()
    torch.cuda.empty_cache()


    # Evaluation --------------------------------------------------------------

    # reload tokenizer 
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    # load the base model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,  ###TODO: 다른 옵션 테스트 
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config = bnb_config,
        device_map="auto"
    )

    # add trained adapter to the pre-trained llama model
    model.load_adapter(args.output_dir, adapter_name="adapter")
    model.enable_adapters() 

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    # text-generation을 위해 huggingface pipeline으로 모델 감싸기
    model_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    # inference on validation set
    conversation_validation_samples = conversation_validation_samples.map(lambda samples: get_base_and_tuned_bulk_predictions(pipe=model_pipe, samples=samples), batched=True, batch_size=5)

    # save the result
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


    # load metrics
    bertscore = load("bertscore")
    em_f1_score = load("squad")

    # score calculation
    bert_predictions = conversation_validation_samples['predicted_answer']
    bert_references = conversation_validation_samples['answer']

    ex_predictions = [{'id': sample['id'], 'prediction_text': sample['predicted_answer']} for sample in conversation_validation_samples]
    ex_references = [{'id': sample['id'], 'answers': [{'text': sample['answers']['text'][0], 'answer_start': sample['answers']['answer_start'][0]}]} for sample in conversation_validation_samples]

    trained_validation_bert_score = bertscore.compute(predictions=bert_predictions, references=bert_references, lang="kr", model_type=args.bert_model, device="cuda:0")
    tuned_exact_match_score = em_f1_score.compute(predictions=ex_predictions, references=ex_references)
    tuned_averages = {
        key: sum(trained_validation_bert_score[key])/len(trained_validation_bert_score[key]) for key in ['precision', 'recall', 'f1']
    }
    tuned_averages['exact_match'] = tuned_exact_match_score['exact_match']
    tuned_averages['squad_f1'] = tuned_exact_match_score['f1']
    logger.info(tuned_averages)


if __name__ == "__main__":
	main()