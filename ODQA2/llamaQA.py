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
        with open(self.args.prompt_path, "r") as f:
            self.prompt_template = f.read()
        self.tokenizer = None 
        self.model = None
        self.model_pipe = None

    def load_model(self):
        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token

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
        self.model = prepare_model_for_kbit_training(self.model)
        self.model.config.use_cache = False

    # Llama conversation format으로 데이터 변환 
    def convert_squad_sample_to_llama_conversation(self, sample):
        question = sample['question']
        context = sample['context']
        answers = sample['answers']['text']
        if len(answers) == 0 :
            answer = "The context does not provide an answer..."
        else:
            answer = sample['answers']['text'][0]

        messages = [
            {"role": "system", "content": self.prompt_template.format(context=context)},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ]
        sample_conversation = self.tokenizer.apply_chat_template(messages, tokenize=False)
        return {"text": sample_conversation, "messages": messages, "answer": answer}

    def get_base_and_tuned_bulk_predictions(self, samples):
        bulk_messages = [i[:-1] for i in samples['messages']]
        responses = self.model_pipe(bulk_messages, max_new_tokens=64, batch_size=len(samples), do_sample=False, top_p=None)
        responses = [i[0]['generated_text'][-1]['content'] for i in responses]
        return {"predicted_answer": responses}

    
    def predict(self, datasets, save_path):
        # text-generation을 위해 huggingface pipeline으로 모델 감싸기
        self.model_pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)
        
        # 데이터 형태 변환 및 추론
        conversation_datasets = datasets.map(self.convert_squad_sample_to_llama_conversation)
        conversation_datasets = conversation_datasets.map(self.get_base_and_tuned_bulk_predictions, batched=True, batch_size=5)

        # Metric을 구할 수 있도록 Format을 맞춰줌
        formatted_predictions = [{"id": sample["id"], "prediction_text": sample["predicted_answer"]} for sample in conversation_datasets]

        # prediction 결과 저장
        test_output_file = save_path + "/predictions.json"
        with open(test_output_file, "w", encoding="utf-8") as f:
            json.dump(formatted_predictions, f, ensure_ascii=False, indent=4)

        return formatted_predictions