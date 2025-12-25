import os
os.environ['TRANSFORMERS_NO_TORCH_LOAD_CHECK'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from peft import PeftModel
import argparse
import os

parser = argparse.ArgumentParser(description='Сравнение моделей: обычная, дообученная на малой/большой выборке')
parser.add_argument('--model_name', type=str, required=True, help='Путь к базовой модели (из HF или локальный)')
parser.add_argument('--lora_version', type=str, required=True, choices=['none', 'small', 'full'], help='Версия для теста: none, small, full')
parser.add_argument('--test_file', type=str, required=True, help='Файл с тестовыми вопросами (test_prompts.txt)')
parser.add_argument('--test_seed', type=int, required=False, default=42, help='Seed для воспроизводимости (по умолчанию: 42)')

args = parser.parse_args()

if args.test_seed is not None:
    set_seed(args.test_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.test_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def read_prompts(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        prompts = [line.strip() for line in lines if line.strip()]
    return prompts

def generate_answer(model, tokenizer, model_type, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=80,
        do_sample=True,
        temperature=0.8,
        top_p=0.92,
        top_k=40,
        repetition_penalty=1.15,
        no_repeat_ngram_size=3,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = full_text[len(prompt):].strip() if full_text.startswith(prompt) else full_text
    
    print(f"\n{model_type}: '{prompt}'")
    print(f"Ответ: {answer}")

version_to_path = {
    'none': None,                   
    'small': "./vaz_lora_small",
    'full': "./vaz_lora_full" 
}
lora_path = version_to_path[args.lora_version]

version_to_description = {
    'none': 'ОБЫЧНАЯ (предобученная) модель',
    'small': 'ДООБУЧЕННАЯ модель (малый датасет)',
    'full': 'ДООБУЧЕННАЯ модель (полный датасет)'
}
model_description = version_to_description[args.lora_version]

print(f"\n{'='*60}")
print(f"ТЕСТ: {model_description}")
print(f"Базовая модель: {args.model_name}")
if lora_path:
    print(f"Адаптер LoRA: {lora_path}")
print(f"Seed: {args.test_seed}")
print(f"{'='*60}")

tokenizer = AutoTokenizer.from_pretrained(args.model_name)

model = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

if lora_path and os.path.exists(lora_path) and os.path.exists(os.path.join(lora_path, "adapter_config.json")):
    model = PeftModel.from_pretrained(model, lora_path)
    print(f"✅ Загружен LoRA адаптер из {lora_path}")
    has_lora = True
else:
    if lora_path:
        print(f"⚠️  Адаптер '{lora_path}' не найден, используется базовая модель")
    has_lora = False

prompts = read_prompts(args.test_file)

for prompt in prompts:
    generate_answer(model, tokenizer, model_description, prompt)

print(f"\n{'='*60}")
print(f"Тестирование '{model_description}' завершено.")
print(f"{'='*60}")