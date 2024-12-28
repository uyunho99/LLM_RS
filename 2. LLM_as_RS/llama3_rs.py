import subprocess
import os
import tqdm as notebook_tqdm

from sklearn.model_selection import train_test_split
import datasets
import pickle

import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, pipeline, logging, Trainer

from peft import LoraConfig
from trl import SFTTrainer

subprocess.run(["huggingface-cli", "login", "--token", "[your token]"])

# Data
with open(file='/data/log-data-2024/prompt_data.pickle', mode='rb') as f:
    data = pickle.load(f)

data1 = data[:100]
# Model
torch_dtype = torch.float16
    
model_ckpt = "meta-llama/Meta-Llama-3.1-8B-Instruct"

bnb_config = BitsAndBytesConfig(
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch_dtype,
    bnb_4bit_use_double_quant=False
)

tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
tokenizer.pad_token = tokenizer.eos_token

print(tokenizer.pad_token, tokenizer.pad_token_id)
print(tokenizer.eos_token, tokenizer.eos_token_id)

model = AutoModelForCausalLM.from_pretrained(
    model_ckpt,
    low_cpu_mem_usage=True,
    quantization_config=bnb_config,
    torch_dtype=torch_dtype,
    device_map='auto'
)

model.config.use_cache = False

# Dataset
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

train_dataset = datasets.Dataset.from_dict({key: [item[key] for item in train_data] for key in train_data[0]})
test_dataset = datasets.Dataset.from_dict({key: [item[key] for item in test_data] for key in test_data[0]})

# Tokenizing

def tokenize_function(examples):
    inputs = tokenizer(examples["prompt"], padding="max_length", truncation=True, max_length=200) # 200까지 줄여도 될듯
    outputs = tokenizer(examples["completion"], padding="max_length", truncation=True, max_length=20) # 20까지 줄여도 될듯
    inputs["labels"] = outputs["input_ids"]
    return inputs

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Peft

from peft import prepare_model_for_kbit_training

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

from peft import LoraConfig, get_peft_model

peft_config = LoraConfig(
    r=16, # attention dim
    lora_alpha=32, # LoRA 스케일링 파라미터
    # target_modules=['query_key_value'],
    # target_modules=['q_proj','k_proj'],
    # target_modules=['q_proj','k_proj','v_proj','o_proj'], # LoRA 적용할 모듈
    lora_dropout=0.1, # LoRA 드롭아웃
    bias="none",
    task_type="CAUSAL_LM",
)

peft_config.inference_mode = False
model = get_peft_model(model, peft_config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Trainer
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=3e-4,
    warmup_steps=100,
    weight_decay=0.01,
    max_grad_norm=1.0,
    # fp16=True,
    # eval_steps=3000,
    logging_steps=10,
    logging_strategy='steps',
    optim="paged_adamw_8bit",
    overwrite_output_dir=True,
    save_strategy="epoch",
    # save_steps=3000,
    save_total_limit=3,
    load_best_model_at_end=True,
    fp16=True,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=256,
    tokenizer=tokenizer,
    args=training_args,
    packing=False,
)

trainer.train()

# Saving
trainer.save_model(output_dir="./results/best_model")
tokenizer.save_pretrained("./results/best_model")