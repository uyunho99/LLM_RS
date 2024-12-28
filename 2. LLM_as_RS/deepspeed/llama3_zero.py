from transformers import BitsAndBytesConfig, TrainingArguments, AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer
from peft import LoraConfig, prepare_model_for_kbit_training
from accelerate import Accelerator, DeepSpeedPlugin
import torch
import datasets
import pickle
from sklearn.model_selection import train_test_split

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_storage=torch.float16,
)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

peft_config.inference_mode = False

ds_config_dict = {
    "zero3": {
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True
            },
            "offload_param": {
                "device": "cpu",
                "pin_memory": True
            },
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": 5e5,
            "stage3_prefetch_bucket_size": 5e5,
            "stage3_param_persistence_threshold": 5e5
        }
    }
}

deepspeed_plugin = DeepSpeedPlugin(
    hf_ds_config=ds_config_dict["zero3"],
    gradient_accumulation_steps=4,
    gradient_clipping=1.0,
    zero_stage=3,
    zero3_save_16bit_model=True,
    zero3_init_flag=True,
)

accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin, mixed_precision="fp16")

base_model = "meta-llama/Meta-Llama-3.1-8B-Instruct"
new_model = "LLM_RS_zero3"
tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
)
model.config.use_cache = False
model = prepare_model_for_kbit_training(model)

with open('/data/log-data-2024/prompt_data.pickle', 'rb') as f:
    data = pickle.load(f)

data_1 = data[:100]
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

train_dataset = datasets.Dataset.from_dict({key: [item[key] for item in train_data] for key in train_data[0]})
test_dataset = datasets.Dataset.from_dict({key: [item[key] for item in test_data] for key in test_data[0]})

def tokenize_function(examples):
    full_text = [p + c for p, c in zip(examples["prompt"], examples["completion"])]
    tokenized = tokenizer(full_text, padding="max_length", truncation=True, max_length=210)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    learning_rate=3e-4,
    warmup_steps=100,
    weight_decay=0.01,
    max_grad_norm=1.0,
    logging_steps=10,
    logging_strategy='steps',
    # optim="paged_adamw_8bit",
    overwrite_output_dir=True,
    save_strategy="epoch",
    save_total_limit=3,
    load_best_model_at_end=True,
    gradient_accumulation_steps=4,
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
trainer.save_model("./saved_model")