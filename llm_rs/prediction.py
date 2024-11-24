import pickle
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from tqdm import tqdm
from collections import Counter
import re

with open('/data/log-data-2024/2.sequence_generate_ksc/data/prompt_final_prediction_20241022.pickle', 'rb') as f:
    prompts = pickle.load(f)

# Modify prompts
for i in range(len(prompts)):
    prompts[i] += " Completion:"

# def replace_prompt_in_list(prompt_list, old_text, new_text):
#     for item in prompt_list:
#         if 'prompt' in item and old_text in item['prompt']:
#             item['prompt'] = item['prompt'].replace(old_text, new_text)
#     return prompt_list

# new_prompt = replace_prompt_in_list(
#     prompt_list=prompts,
#     old_text="You are a recommender system. Based on the contents that the user has previously chosen, recommend a proper TV content that the user may prefer. Output the title of the recommended TV content which is listed in the dictionary. Previously chosen contents",
#     new_text="You are a recommender system. Based on the contents that the user has previously chosen, recommend a proper TV content that the user may prefer. Output the number and title of the recommended TV content which is listed in the dictionary like this formation '[# of Content]. [Content Title]'. Previously chosen contents"
# )

def replace_prompt_in_list_1(prompt_list, old_text, new_text):
    for i in range(len(prompt_list)):
        prompt_list[i] = prompt_list[i].replace(old_text, new_text)
    return prompt_list

new_prompt = replace_prompt_in_list_1(
    prompt_list=prompts,
    old_text="You are a recommender system. Based on the contents that the user has previously chosen, recommend a proper TV content that the user may prefer. Output the title of the recommended TV content which is listed in the dictionary.",
    new_text="You are a recommender system. Based on the contents that the user has previously chosen, recommend a proper TV content that the user may prefer. Output the number and title of the recommended TV content which is listed in the dictionary like this formation '[# of Content]. [Content Title]'."
)

# numb = 10225

new_prompt_1 = new_prompt[6105:]
prompts_dict = [{'prompt': prompt} for prompt in new_prompt_1]

# Model configuration parameters (outside the loop)
model_path = "/data/log-data-2024/yh/LLM_RS/results/best_model_1106"

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

# Load tokenizer outside the loop
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prediction function
def generate_prediction(prompts, model, tokenizer):
    predictions = []
    for item in tqdm(prompts, desc="Generating predictions"):
        prompt_text = item['prompt']
        inputs = tokenizer(prompt_text, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=4,
                num_beams=25,
                num_return_sequences=25,
                do_sample=False,
            )
        predictions_batch = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        predictions.append(predictions_batch)
    return predictions

# Function to extract top numbers
def extract_top_numbers(predicted_sequences):
    top_numbers_list = []
    for sequences in predicted_sequences:
        # Extract all numbers
        all_numbers = []
        for sequence in sequences:
            match = re.search(r'Completion:\s*(\d+)', sequence)
            if match:
                all_numbers.append(match.group(1))
        # Count numbers and get top 10
        number_counts = Counter(all_numbers)
        most_common_numbers = number_counts.most_common()
        # Sort numbers based on count and first occurrence
        sorted_numbers = sorted(most_common_numbers, key=lambda x: (-x[1], all_numbers.index(x[0])))
        top_10_numbers = [num[0] for num in sorted_numbers[:10]]
        top_numbers_list.append(top_10_numbers)
    return top_numbers_list

# Process prompts in batches of 100
batch_size = 5
total_prompts = len(prompts_dict)

for i in range(0, total_prompts, batch_size):
    print(f"Processing prompts from index {i} to {min(i + batch_size - 1, total_prompts - 1)}")
    prompts_batch = prompts_dict[i:i + batch_size]
    
    # Load model inside the loop
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    
    # Generate predictions
    predicted_sequences = generate_prediction(prompts_batch, model, tokenizer)
    
    # Unload model to free up GPU memory
    del model
    torch.cuda.empty_cache()
    
    # Extract top numbers
    top_numbers = extract_top_numbers(predicted_sequences)
    
    # Save results to DataFrame
    df_top_numbers = pd.DataFrame(top_numbers)
    df_top_numbers.insert(0, 'prompt_id', range(i + 6105, i + len(df_top_numbers) + 6105))
    
    # Save to CSV
    df_top_numbers.to_csv(f'./top_numbers_{i + 6105}_{i + len(df_top_numbers) - 1 + 6105}.csv', index=False)