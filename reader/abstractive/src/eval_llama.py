# reference: https://medium.com/@coldstart_coder/fine-tuning-llama-3-2-11b-for-question-answering-435c28bb57c1


# Data Preparation ------------------------------------------

from datasets import load_from_disk

# 데이터 불러오기
data_path = "/data/ephemeral/home/datasets/v0.0.2"
data = load_from_disk(data_path)
train_data = data['train']
valid_data = data['validation']

# 모델 및 토크나이저 로드
from transformers import AutoTokenizer
model_name = "meta-llama/Llama-3.1-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

# Llama conversation format으로 데이터 변환 
def convert_squad_sample_to_llama_conversation(sample):
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
    instruction_prompt_template = '''
    You are a helpful assistant tasked with extracting passages that answer users questions from a given context. Output exact passages word for word that answer the users question. Do not output any other text other than passages in the context passage. Output the minimal amount to answer the question, for example only 2-3 words from the passage. If you cannot find the answer in the context passage output 'The context does not provide an answer...'

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

conversation_training_samples = train_data.map(convert_squad_sample_to_llama_conversation)
conversation_validation_samples = valid_data.map(convert_squad_sample_to_llama_conversation)



# Loading The Model and Trained Adapters ------------------------------------------
# 저장된 adapter 가중치와, base model 같이 불러오면 됨

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
import torch
final_model_path = "./model/final_model"
model_name = "meta-llama/Llama-3.1-8B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'

# first we'll load in the base model
# to help save on gpu space and run this a bit faster we'll load the model in 4bit
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="cuda")

# next we'll add our trained adapter to the model
model.load_adapter(final_model_path, adapter_name="adapter")

# now using enable_adapters and disable_adapters we can choose
# if we want to run inference on the model itself or have it be
# influenced by our newly trained weights
model.enable_adapters()

# also make sure we set the pad token, and for good measure turn off caching
model.config.pad_token_id = tokenizer.pad_token_id
model.config.use_cache = False

# last we wrap the model in a hugging face pipeline for text-generation
# this helps streamline our inference code a bit by managing the inputs/outputs for us
model_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)


# Evaluation ------------------------------------------
from evaluate import load

# there are several bert models we can use as a part of bert score
bert_model = "klue/bert-base"
bertscore = load("bertscore")

exact_match_metric = load("exact_match")


# Helper function that will take in a pipeline loaded with our llm for question answer
# and a list of samples to run inference on, it will return just the responses for each sample
def get_bulk_predictions(pipe, samples):
    responses = pipe(samples, max_new_tokens=512, batch_size=len(samples), do_sample=False)
    responses = [i[0]['generated_text'][-1]['content'] for i in responses]
    return responses

# helper function that will take in a list of samples and run inference for both
# our tuned and base model and return the results
def get_base_and_tuned_bulk_predictions(samples):
    bulk_messages = [i[:-1] for i in samples['messages']]

    # first we enable the adapters in our model, so that inference with our pipeline
    # will be influenced by our trained weights.
    # then get the responses for our tuned version of the model.
    model.enable_adapters()
    trained_responses = get_bulk_predictions(model_pipe, bulk_messages)

    # next turn off adapters and get the inference using just the base model weights.
    model.disable_adapters()
    base_responses = get_bulk_predictions(model_pipe, bulk_messages)

    # now return the base model predictions and the tuned model predictions
    return {"base_prediction": base_responses, "trained_prediction": trained_responses}

# run inference on our validation set
conversation_validation_samples = conversation_validation_samples.map(get_base_and_tuned_bulk_predictions, batched=True, batch_size=20)


# calculate the scores for the base model
base_predictions = conversation_validation_samples['base_prediction']
answers = conversation_validation_samples['answer']
base_validation_bert_score = bertscore.compute(predictions=base_predictions, references=answers, lang="en", model_type=bert_model, device="cuda:0")
baseline_exact_match_score = exact_match_metric.compute(predictions=base_predictions, references=answers)
baseline_averages = {
    key: sum(base_validation_bert_score[key])/len(base_validation_bert_score[key]) for key in ['precision', 'recall', 'f1']
}
baseline_averages['exact_match'] = sum(baseline_exact_match_score.values())/len(baseline_exact_match_score.values())

print(baseline_averages)

# do the same calculation for the tuned model
trained_predictions = conversation_validation_samples['trained_prediction']
answers = conversation_validation_samples['answer']
trained_validation_bert_score = bertscore.compute(predictions=trained_predictions, references=answers, lang="en", model_type=bert_model, device="cuda:0")
tuned_exact_match_score = exact_match_metric.compute(predictions=trained_predictions, references=answers)
tuned_averages = {
    key: sum(trained_validation_bert_score[key])/len(trained_validation_bert_score[key]) for key in ['precision', 'recall', 'f1']
}

tuned_averages['exact_match'] = sum(tuned_exact_match_score.values())/len(tuned_exact_match_score.values())
print(tuned_averages)