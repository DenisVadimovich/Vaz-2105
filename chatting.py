import os
os.environ['TRANSFORMERS_NO_TORCH_LOAD_CHECK'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from peft import PeftModel
import argparse
import signal
import sys

def signal_handler(sig, frame):
    print("\n\nДиалог завершен. Выход...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

parser = argparse.ArgumentParser(description='Чат с моделью: обычная или дообученная с LoRA')
parser.add_argument('--model_name', type=str, required=True, help='Путь к базовой модели')
parser.add_argument('--lora_version', type=str, required=True, choices=['none', 'small', 'full'], 
                    help='Версия модели: none (обычная), small (малый датасет), full (полный датасет)')
parser.add_argument('--seed', type=int, required=False, default=None, 
                    help='Seed для воспроизводимости (по умолчанию: None - случайный)')

args = parser.parse_args()

if args.seed is not None:
    set_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class ChatBot:
    def __init__(self, model, tokenizer, model_description):
        self.model = model
        self.tokenizer = tokenizer
        self.model_description = model_description
        self.conversation_history = []
        
    def add_to_history(self, role, text):
        self.conversation_history.append({"role": role, "text": text})
    
    def get_prompt_with_history(self, user_input):
        prompt = ""
        
        for message in self.conversation_history[-10:]:  
            if message["role"] == "user":
                prompt += f"Вопрос: {message['text']}\n"
            else:
                prompt += f"Ответ: {message['text']}\n"
        
        prompt += f"Вопрос: {user_input}\nОтвет:"
        return prompt
    
    def generate_response(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=80,
                do_sample=True,
                temperature=0.8,
                top_p=0.92,
                top_k=40,
                repetition_penalty=1.15,
                no_repeat_ngram_size=3,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "Ответ:" in full_text:
            response = full_text.split("Ответ:")[-1].strip()
        else:
            response = full_text[len(prompt):].strip()
        
        return response
    
    def chat_loop(self):
        print("\n" + "="*60)
        print(f"Чат с моделью: {self.model_description}")
        print(f"Seed: {args.seed if args.seed is not None else 'случайный'}")
        print("Вопросы: вводите текст, 'история' для просмотра, 'очистить' для очистки истории")
        print("Выход: 'выход', 'exit', 'quit' или Ctrl+C")
        print("="*60 + "\n")
        
        while True:
            try:
                user_input = input("Вопрос: ").strip()
                
                if user_input.lower() in ['выход', 'exit', 'quit']:
                    print("Завершение диалога...")
                    break
                
                if user_input.lower() == 'история':
                    self.show_history()
                    continue
                
                if user_input.lower() == 'очистить':
                    self.clear_history()
                    print("История диалога очищена.")
                    continue
                
                if not user_input:
                    continue

                prompt_with_history = self.get_prompt_with_history(user_input)

                self.add_to_history("user", user_input)

                print("Модель генерирует ответ...", end="", flush=True)
                response = self.generate_response(prompt_with_history)
                print("\r" + " " * 40 + "\r", end="")

                print(f"Ответ: {response}\n")
                self.add_to_history("assistant", response)
                
            except KeyboardInterrupt:
                print("\n\nДиалог прерван пользователем.")
                break
            except Exception as e:
                print(f"\nОшибка при генерации: {e}")
                continue
    
    def show_history(self):
        if not self.conversation_history:
            print("История диалога пуста.\n")
            return
        
        print("\n" + "-"*40)
        print("ИСТОРИЯ ДИАЛОГА:")
        print("-"*40)
        for i, message in enumerate(self.conversation_history, 1):
            role = "Пользователь" if message["role"] == "user" else "Модель"
            print(f"{i}. {role}: {message['text']}")
        print("-"*40 + "\n")
    
    def clear_history(self):
        self.conversation_history = []

def main():
    version_to_path = {
        'none': None,     
        'small': "./vaz_lora_small",      
        'full': "./vaz_lora_full"        
    }
    
    version_to_description = {
        'none': 'ОБЫЧНАЯ (предобученная) модель',
        'small': 'ДООБУЧЕННАЯ модель (малый датасет)',
        'full': 'ДООБУЧЕННАЯ модель (полный датасет)'
    }
    
    lora_path = version_to_path[args.lora_version]
    model_description = version_to_description[args.lora_version]
    
    print(f"\n{'='*60}")
    print(f"ЗАГРУЗКА МОДЕЛИ ДЛЯ ЧАТА")
    print(f"Тип: {model_description}")
    print(f"Базовая модель: {args.model_name}")
    if lora_path:
        print(f"Адаптер LoRA: {lora_path}")
    print(f"Seed: {args.seed if args.seed is not None else 'случайный'}")
    print(f"{'='*60}")
    print("Загрузка токенизатора...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    
    print(f"Загрузка модели на устройство {device}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
        device_map="auto" if device == "cuda" else {"": device},
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )

    if lora_path and os.path.exists(lora_path) and os.path.exists(os.path.join(lora_path, "adapter_config.json")):
        model = PeftModel.from_pretrained(model, lora_path)
        print(f"✅ Загружен LoRA адаптер из {lora_path}")
    else:
        if lora_path:
            print(f"⚠️  Адаптер '{lora_path}' не найден, используется базовая модель")

    model.eval()

    chatbot = ChatBot(model, tokenizer, model_description)
    chatbot.chat_loop()

if __name__ == "__main__":
    main()