import sys
import logging
import json
from box import Box
import argparse
from datasets import load_from_disk
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


class LlamaQA():
    def __init__(self, config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        self.args = Box(config)
        self.tokenizer = None 
        self.model = None
        self.model_pipe = None

    def load_model(self):
        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.chat_template = self.chat_template

        # load the base model
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=self.args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.args.model_name,
            quantization_config = bnb_config,
            device_map="auto"
        )

        # add trained adapter to the pre-trained llama model
        self.model.load_adapter(self.args.output_dir, adapter_name="adapter")
        self.model.enable_adapters() 

        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.use_cache = False

    # Llama conversation format으로 데이터 변환 
    def convert_squad_sample_to_llama_conversation(self, sample):
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
        sample_conversation = self.tokenizer.apply_chat_template(messages)
        return {"text": sample_conversation, "messages": messages, "answer": answer}

    def get_base_and_tuned_bulk_predictions(self, samples):
        bulk_messages = [i[:-1] for i in samples['messages']]
        responses = self.pipe(bulk_messages, max_new_tokens=512, batch_size=len(samples), do_sample=False, top_p=None)
        responses = [i[0]['generated_text'][-1]['content'] for i in responses]
        return {"predicted_answer": responses}

    def predict(self, datasets, save_path):
        # text-generation을 위해 huggingface pipeline으로 모델 감싸기
        self.model_pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)

        with open(self.args.chat_template_path, 'r') as file:
            chat_template = file.read()
        self.tokenizer.chat_template = chat_template

        datasets = load_from_disk(self.args.data_path)
        eval_dataset = datasets['validation']
        test_dataset = datasets['test']

        eval_dataset = eval_dataset.select(range(20))
        test_dataset = test_dataset.select(range(20))

        # Llama 학습을 위한 형태로 데이터 변환
        conversation_validation_samples = eval_dataset.map(self.convert_squad_sample_to_llama_conversation)
        conversation_test_samples = test_dataset.map(self.convert_squad_sample_to_llama_conversation)

        # inference 
        conversation_validation_samples = conversation_validation_samples.map(self.get_base_and_tuned_bulk_predictions, batched=True, batch_size=20)
        conversation_test_samples = conversation_test_samples.map(self.get_base_and_tuned_bulk_predictions, batched=True, batch_size=20)

        # prediction 결과 저장
        valid_result_dict = {}  # id별로 answer와 predicted_answer 저장할 계층형 딕셔너리 생성
        for sample in conversation_validation_samples:
            sample_id = sample['id']
            answer = sample['answer']
            predicted_answer = sample['predicted_answer']
            
            valid_result_dict[sample_id] = {
                'answer': answer,
                'predicted_answer': predicted_answer
            }
        valid_output_file = save_path + "/valid/predictions_with_answer.json"
        with open(valid_output_file, "w", encoding="utf-8") as f:
            json.dump(valid_result_dict, f, ensure_ascii=False, indent=4)

        test_result_dict = {}
        for sample in conversation_test_samples:
            sample_id = sample['id']
            predicted_answer = sample['predicted_answer']
            test_result_dict[sample_id] = predicted_answer
        test_output_file = save_path + "/test/predictions.json"
        with open(test_output_file, "w", encoding="utf-8") as f:
            json.dump(test_result_dict, f, ensure_ascii=False, indent=4)


        # score calculation
        bertscore = load("bertscore")
        em_f1_score = load("squad")

        # score calculation
        bert_predictions = conversation_validation_samples['predicted_answer']
        bert_references = conversation_validation_samples['answer']

        ex_predictions = [{'id': sample['id'], 'prediction_text': sample['predicted_answer']} for sample in conversation_validation_samples]
        ex_references = [{'id': sample['id'], 'answers': [{'text': sample['answers']['text'][0], 'answer_start': sample['answers']['answer_start'][0]}]} for sample in conversation_validation_samples]

        trained_validation_bert_score = bertscore.compute(predictions=bert_predictions, references=bert_references, lang="kr", model_type=self.args.bert_model, device="cuda:0")
        tuned_exact_match_score = em_f1_score.compute(predictions=ex_predictions, references=ex_references)
        tuned_averages = {
            key: sum(trained_validation_bert_score[key])/len(trained_validation_bert_score[key]) for key in ['precision', 'recall', 'f1']
        }
        tuned_averages['exact_match'] = tuned_exact_match_score['exact_match']
        tuned_averages['squad_f1'] = tuned_exact_match_score['f1']
        logger.info(tuned_averages)
