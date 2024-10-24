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

        with open(self.args.chat_template_path, 'r') as file:
            chat_template = file.read()
        self.tokenizer.chat_template = chat_template

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
        responses = self.model_pipe(bulk_messages, max_new_tokens=512, batch_size=len(samples), do_sample=False, top_p=None)
        responses = [i[0]['generated_text'][-1]['content'] for i in responses]
        return {"predicted_answer": responses}

    
    def predict(self, datasets, save_path):
        # text-generation을 위해 huggingface pipeline으로 모델 감싸기
        self.model_pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)

        with open(self.args.chat_template_path, 'r') as file:
            chat_template = file.read()
        self.tokenizer.chat_template = chat_template

        # 데이터 형태 변환 및 추론
        conversation_datasets = datasets.map(self.convert_squad_sample_to_llama_conversation)
        conversation_datasets = conversation_datasets.map(self.get_base_and_tuned_bulk_predictions, batched=True, batch_size=20)

        # prediction 결과 저장
        test_result_dict = {}
        for sample in conversation_datasets:
            sample_id = sample['id']
            predicted_answer = sample['predicted_answer']
            test_result_dict[sample_id] = predicted_answer
        test_output_file = save_path + "/predictions.json"
        with open(test_output_file, "w", encoding="utf-8") as f:
            json.dump(test_result_dict, f, ensure_ascii=False, indent=4)

        return test_result_dict