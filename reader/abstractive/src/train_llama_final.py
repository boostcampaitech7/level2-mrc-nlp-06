# reference: https://medium.com/@coldstart_coder/fine-tuning-llama-3-2-11b-for-question-answering-435c28bb57c1

# Llama 3.2 11B 모델을 Q-LoRA fine-tuning해서 QA task 접근
# Llama 3.1 8B 

# metric
# - bert-score: 모델 출력과 예상 답변 사이의 의미적 유사성 측정
# - EM, F1-score: 정확한 답변 반환 여부


# Model and Dataset Access
# huggingface-cli login

# Data Preparation ------------------------------------------

from datasets import load_from_disk
import argparse
import logging
import sys
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, pipeline
import torch
from peft import prepare_model_for_kbit_training, LoraConfig
from trl import SFTTrainer
from evaluate import load
import os
from box import Box


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
def convert_squad_sample_to_llama_conversation(sample, tokenizer):
    # get the question and context for this sample
    question = sample['question']
    context = sample['context']

    # some questions can have multiple answers, some none at all,
    # for the case of no answers we'll have the model output that the
    # context does not provide an answer, if it has multiple we'll just take
    # the first answer as the ground truth.
    answers = sample['answers']['text']
    if len(answers) == 0 :
      answer = "The context does not provide an answer..."
    else:
      answer = sample['answers']['text'][0]

    # now we define an initial model prompt defining the task and giving the model the context passage
    # 프롬프트 한국어로 바꾸면 성능 더 좋아질까?
    instruction_prompt_template = '''You are a helpful assistant tasked with extracting passages that answer users questions from a given context. Output exact passages word for word that answer the users question. Do not output any other text other than passages in the context passage. Output the minimal amount to answer the question, for example only 2-3 words from the passage. If you cannot find the answer in the context passage output 'The context does not provide an answer...'

    Context: {context}'''

    # now we'll convert these into a list of messages for our conversation
    messages = [
        {"role": "system", "content": instruction_prompt_template.format(context=context)},
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer}
    ]
    # apply the chat template and return the sample
    # we'll also return the single answer we expect and the list of messages without
    # the chat template in case we need them later.
    sample_conversation = tokenizer.apply_chat_template(messages, tokenize=False)
    return {"text": sample_conversation, "messages": messages, "answer": answer}


# Helper function that will take in a pipeline loaded with our llm for question answer
# and a list of samples to run inference on, it will return just the responses for each sample
# def get_bulk_predictions(pipe, samples):
#     responses = pipe(samples, max_new_tokens=512, batch_size=len(samples), do_sample=False, top_p=None)
#     responses = [i[0]['generated_text'][-1]['content'] for i in responses]
#     return responses

# helper function that will take in a list of samples and run inference for both
# our tuned and base model and return the results
def get_base_and_tuned_bulk_predictions(pipe, samples, model):
    bulk_messages = [i[:-1] for i in samples['messages']]

    # first we enable the adapters in our model, so that inference with our pipeline
    # will be influenced by our trained weights.
    # then get the responses for our tuned version of the model.
    # model.enable_adapters()  # 추가 학습된 가중치 이용해 인퍼런스 (<-> 기본 모델 사용: disable_adapters)
    responses = pipe(bulk_messages, max_new_tokens=512, batch_size=len(samples), do_sample=False, top_p=None)
    responses = [i[0]['generated_text'][-1]['content'] for i in responses]

    # now return the base model predictions and the tuned model predictions
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

    # 데이터 불러오기
    datasets = load_from_disk(args.data_path)
    logger.info(datasets)
    train_dataset = datasets['train']
    eval_dataset = datasets['validation']

    with open(args.chat_template_path, 'r') as file:
        chat_template = file.read()
    
    # 모델 및 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = chat_template

    # Llama 학습을 위한 형태로 데이터 변환
    conversation_training_samples = train_dataset.map(lambda sample: convert_squad_sample_to_llama_conversation(sample, tokenizer))
    conversation_validation_samples = eval_dataset.map(lambda sample: convert_squad_sample_to_llama_conversation(sample, tokenizer))


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
    model.config.pad_token_id = tokenizer.pad_token_id
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
        # the target modules defines what types of layers to add lora adapters too, so in the network
        # any model that have a name in this list will have a lora adapter added to it,
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
        num_train_epochs=3,
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
    torch.cuda.empty_cache()


    # Evaluation --------------------------------------------------------------

    # add trained adapter to the pre-trained llama model
    if not model.is_adapter_loaded("adapter"):
        model.load_adapter(args.output_dir, adapter_name="adapter")
        model.enable_adapters() 

    # text-generation을 위해 huggingface pipeline으로 모델 감싸기
    model_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    # inference on validation set
    conversation_validation_samples = conversation_validation_samples.map(lambda samples: get_base_and_tuned_bulk_predictions(pipe=model_pipe, samples=samples, model=model), batched=True, batch_size=20)

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

    output_file = args.output_dir + "predictions.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=4)

    logger.info(f"Data saved to {output_file}")


    # load metrics
    bertscore = load("bertscore")
    em_f1_score = load("squad")

    # score calculation
    predicted_answer = conversation_validation_samples['predicted_answer']
    answers = conversation_validation_samples['answer']
    trained_validation_bert_score = bertscore.compute(predictions=predicted_answer, references=answers, lang="en", model_type=args.bert_model, device="cuda:0")
    tuned_exact_match_score = em_f1_score.compute(predictions=predicted_answer, references=answers)
    tuned_averages = {
        key: sum(trained_validation_bert_score[key])/len(trained_validation_bert_score[key]) for key in ['precision', 'recall', 'f1']
    }
    tuned_averages['exact_match'] = sum(tuned_exact_match_score['exact_match']) / len(tuned_exact_match_score['exact_match'])
    tuned_averages['squad_f1'] = sum(tuned_exact_match_score['f1']) / len(tuned_exact_match_score['f1'])
    logger.info(tuned_averages)


if __name__ == "__main__":
	main()