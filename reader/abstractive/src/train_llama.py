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


tokenizer.chat_template = """{{- bos_token }}
{%- if custom_tools is defined %}
    {%- set tools = custom_tools %}
{%- endif %}
{%- if not tools_in_user_message is defined %}
    {%- set tools_in_user_message = true %}
{%- endif %}
{%- if not date_string is defined %}
    {%- if strftime_now is defined %}
        {%- set date_string = strftime_now("%d %b %Y") %}
    {%- else %}
        {%- set date_string = "26 Jul 2024" %}
    {%- endif %}
{%- endif %}
{%- if not tools is defined %}
    {%- set tools = none %}
{%- endif %}

{#- This block extracts the system message, so we can slot it into the right place. #}
{%- if messages[0]['role'] == 'system' %}
    {%- set system_message = messages[0]['content']|trim %}
    {%- set messages = messages[1:] %}
{%- else %}
    {%- set system_message = "" %}
{%- endif %}

{#- Find out if there are any images #}
{% set image_ns = namespace(has_images=false) %}      
{%- for message in messages %}
    {%- for content in message['content'] %}
        {%- if content['type'] == 'image' %}
            {%- set image_ns.has_images = true %}
        {%- endif %}
    {%- endfor %}
{%- endfor %}

{#- Error out if there are images and system message #}
{%- if image_ns.has_images and not system_message == "" %}
    {{- raise_exception("Prompting with images is incompatible with system messages.") }}
{%- endif %}

{#- System message if there are no images #}
{%- if not image_ns.has_images %}
    {{- "<|start_header_id|>system<|end_header_id|>\n\n" }}
    {%- if tools is not none %}
        {{- "Environment: ipython\n" }}
    {%- endif %}
    {{- "Cutting Knowledge Date: December 2023\n" }}
    {{- "Today Date: " + date_string + "\n\n" }}
    {%- if tools is not none and not tools_in_user_message %}
        {{- "You have access to the following functions. To call a function, please respond with JSON for a function call." }}
        {{- 'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.' }}
        {{- "Do not use variables.\n\n" }}
        {%- for t in tools %}
            {{- t | tojson(indent=4) }}
            {{- "\n\n" }}
        {%- endfor %}
    {%- endif %}
    {{- system_message }}
    {{- "<|eot_id|>" }}
{%- endif %}

{#- Custom tools are passed in a user message with some extra guidance #}
{%- if tools_in_user_message and not tools is none %}
    {#- Extract the first user message so we can plug it in here #}
    {%- if messages | length != 0 %}
        {%- set first_user_message = messages[0]['content']|trim %}
        {%- set messages = messages[1:] %}
    {%- else %}
        {{- raise_exception("Cannot put tools in the first user message when there's no first user message!") }}
{%- endif %}
    {{- '<|start_header_id|>user<|end_header_id|>\n\n' -}}
    {{- "Given the following functions, please respond with a JSON for a function call " }}
    {{- "with its proper arguments that best answers the given prompt.\n\n" }}
    {{- 'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.' }}
    {{- "Do not use variables.\n\n" }}
    {%- for t in tools %}
        {{- t | tojson(indent=4) }}
        {{- "\n\n" }}
    {%- endfor %}
    {{- first_user_message + "<|eot_id|>"}}
{%- endif %}

{%- for message in messages %}
    {%- if not (message.role == 'ipython' or message.role == 'tool' or 'tool_calls' in message) %}
    {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' }}
        {%- if message['content'] is string %}
            {{- message['content'] }}
        {%- else %}
            {%- for content in message['content'] %}
                {%- if content['type'] == 'image' %}
                    {{- '<|image|>' }}
                {%- elif content['type'] == 'text' %}
                    {{- content['text'] }}
                {%- endif %}
            {%- endfor %}
        {%- endif %}
        {{- '<|eot_id|>' }}
    {%- elif 'tool_calls' in message %}
        {%- if not message.tool_calls|length == 1 %}
            {{- raise_exception("This model only supports single tool-calls at once!") }}
        {%- endif %}
        {%- set tool_call = message.tool_calls[0].function %}
        {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' -}}
        {{- '{"name": "' + tool_call.name + '", ' }}
        {{- '"parameters": ' }}
        {{- tool_call.arguments | tojson }}
        {{- "}" }}
        {{- "<|eot_id|>" }}
    {%- elif message.role == "tool" or message.role == "ipython" %}
        {{- "<|start_header_id|>ipython<|end_header_id|>\n\n" }}
        {%- if message.content is mapping or message.content is iterable %}
            {{- message.content | tojson }}
        {%- else %}
            {{- message.content }}
        {%- endif %}
        {{- "<|eot_id|>" }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
{%- endif %}"""


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

conversation_training_samples = train_data.map(convert_squad_sample_to_llama_conversation)
conversation_validation_samples = valid_data.map(convert_squad_sample_to_llama_conversation)


# Model Preparation Preparation ------------------------------------------

# load the base model with 4-bit precision to help save on the gpu overhead
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

# load the base model with 4-bit precision to help save on the gpu overhead.
# setup our config for the LoRA weights
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',  ###TODO: 다른 옵션 테스트 
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config = bnb_config,
    device_map="auto"
)
model.config.pad_token_id = tokenizer.pad_token_id


from peft import prepare_model_for_kbit_training
model = prepare_model_for_kbit_training(model)
model.config.pad_token_id = tokenizer.pad_token_id
model.config.use_cache = False   # 학습 중이고, output이 변경될 것으로 예상되므로 


# PEFT config 정의
from peft import LoraConfig
# rank defines the rank of the adapter matrix,
# the higher the rank, the more complex the task it's trying to learn
rank = 128

# the alpha is a scaling factor hyper parameter, basically controls how much our
# adapter will influence the models output, the higher this value
# the more our adapter will overpower the original model weights.
# there is a lot of advice out there for what the alpha value should be
# keeping the alpha at around 2x of what the rank is works for this notebook
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

# Training ------------------------------------------
# supervised training

from transformers import TrainingArguments
from trl import SFTTrainer

model_checkpoint_path = "./model/checkpoints"

# an important note is that the loss function isn't defined here,
# it's instead stored as a model parameter for models in hf,
# in the case of llama it is cross entropy loss

# first define some training arguments
training_arguments = TrainingArguments(
    output_dir=model_checkpoint_path,
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
    num_train_epochs=1,
    warmup_ratio=0.1,
    load_best_model_at_end = True,
    overwrite_output_dir = True,
    lr_scheduler_type='linear',# and set our learning rate decay
)

# now that we have our arguments, we'll use that to create our trainer,
# passing in the model, dataset, peft config, tokenizer, ect
trainer = SFTTrainer(
    model=model,
    train_dataset=conversation_training_samples,
    eval_dataset=conversation_validation_samples,
    peft_config=peft_config,
    dataset_text_field='text', # datasets always has samples in a dictionary, so we need to specify what key to reference when training
    max_seq_length=64, # specify how many tokens to generate per training, this is just so it doesn't generate for forever especially for shorter samples
    tokenizer=tokenizer,
    args=training_arguments
)

trainer.model.print_trainable_parameters()

# 학습 시작 전, validation set에 대해 어느 정도의 성능 보이는지 확인
initial_eval_values = trainer.evaluate()
print(initial_eval_values)
initial_eval_loss = initial_eval_values['eval_loss']

# 학습!!!!!!
trainer.train()

# saving the final model weights
# adapter weight만 학습했으므로 이 adapter만 저장됨
final_model_path = "./model/final_model"
trainer.save_model(final_model_path)
torch.cuda.empty_cache()