import argparse
import json
import os
os.environ['TRANSFORMERS_NO_TORCH_LOAD_CHECK'] = '1'
from dataclasses import dataclass
from typing import List, Dict

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)

from peft import LoraConfig, get_peft_model, TaskType, PeftConfig, PeftModel

def read_json_or_jsonl(path: str) -> List[Dict[str, str]]:
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read().strip()
        if not text:
            return []
        try:
            obj = json.loads(text)
            if isinstance(obj, list):
                return obj
        except Exception:
            pass
        items = []
        with open(path, 'r', encoding='utf-8') as fr:
            for line in fr:
                line = line.strip()
                if not line:
                    continue
                items.append(json.loads(line))
        return items


def get_default_lora_targets(model_name: str) -> List[str]:
    mn = model_name.lower()
    if 'qwen' in mn:
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    if 'llama' in mn or 'alpaca' in mn:
        return ["q_proj", "k_proj", "v_proj", "o_proj"]
    if 'gpt2' in mn or 'gpt' in mn or 'dialo' in mn:
        return ["c_attn", "c_proj", "c_fc", "c_ffn"]
    return ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']

def prepare_dataset(items: List[Dict[str, str]], tokenizer: AutoTokenizer, max_length: int = 512):

    records = []

    if tokenizer.eos_token_id is not None:
        sep_token_id = tokenizer.eos_token_id
    else:
        sep_token_id = None

    for it in items:
        prompt = it.get('prompt', '') or ''
        response = it.get('response', '') or ''

        enc_prompt = tokenizer(prompt, add_special_tokens=False)
        enc_resp = tokenizer(response, add_special_tokens=False)

        prompt_ids = enc_prompt["input_ids"]
        resp_ids = enc_resp["input_ids"]

        if sep_token_id is not None:
            input_ids = prompt_ids + [sep_token_id] + resp_ids
            prompt_len = len(prompt_ids) + 1
        else:
            sep_enc = tokenizer("\n", add_special_tokens=False)
            sep_ids = sep_enc["input_ids"]
            input_ids = prompt_ids + sep_ids + resp_ids
            prompt_len = len(prompt_ids) + len(sep_ids)

        input_ids = input_ids[-max_length:] if len(input_ids) > max_length else input_ids

        if prompt_len > len(input_ids):
            labels = [-100] * len(input_ids)
        else:
            labels = [-100] * prompt_len + input_ids[prompt_len:]
            if len(labels) < len(input_ids):
                labels = labels + [-100] * (len(input_ids) - len(labels))
            elif len(labels) > len(input_ids):
                labels = labels[:len(input_ids)]

        attention_mask = [1] * len(input_ids)

        records.append({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        })

    ds = Dataset.from_list(records)
    return ds


@dataclass
class DataCollatorForCausalLMWithLabels:
    tokenizer: AutoTokenizer
    max_length: int = 512

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
        attention_mask = [torch.tensor(f["attention_mask"], dtype=torch.long) for f in features]
        labels = [torch.tensor(f["labels"], dtype=torch.long) for f in features]

        max_len = max(x.size(0) for x in input_ids)

        def pad(tensor, pad_value):
            return torch.nn.functional.pad(
                tensor,
                (0, max_len - tensor.size(0)),
                value=pad_value
            )

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 0

        input_ids = torch.stack([pad(x, pad_id) for x in input_ids])
        attention_mask = torch.stack([pad(x, 0) for x in attention_mask])
        labels = torch.stack([pad(x, -100) for x in labels])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


def main():
    parser = argparse.ArgumentParser(description='Fine-tune causal LM with LoRA (PEFT)')

    parser.add_argument('--model_name', type=str, default='C:/LLM_Project/rugpt3small_local', help='Название модели(из HF)')
    parser.add_argument('--data_path', type=str, required=True, help='Путь к data.json или data.jsonl')
    parser.add_argument('--output_dir', type=str, default='./fine-tuned-lora', help='Куда сохранить модель/адаптер')
    parser.add_argument('--per_device_train_batch_size', type=int, default=4)
    parser.add_argument('--num_train_epochs', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--use_lora', action='store_true')
    parser.add_argument('--lora_r', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    print(f"Загрузка токенизатора и модели: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, 
        torch_dtype=torch.float16,
        device_map="cpu"     
    )   
    try:
        model.gradient_checkpointing_enable()
    except Exception:
        pass
    model.config.use_cache = False
    #print(f"GPU память доступна: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} ГБ")
    if args.use_lora:
        target_modules = get_default_lora_targets(args.model_name)
        print(f"Используем target_modules для LoRA: {target_modules}")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    items = read_json_or_jsonl(args.data_path)
    if not items:
        raise SystemExit("Данные пусты или не удалось прочитать файл")

    ds = prepare_dataset(items, tokenizer, max_length=args.max_length)
    data_collator = DataCollatorForCausalLMWithLabels(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_strategy="no", 
        save_total_limit=1,
        fp16=True,
        gradient_accumulation_steps=2,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=data_collator,
    )

    print("Начинаем обучение...")
    trainer.train()

    print("Сохраняем результат...")
    os.makedirs(args.output_dir, exist_ok=True)

    if args.use_lora:
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print(f"Сохранён LoRA-адаптер и токенизатор в {args.output_dir}")
    else:
        model.save_pretrained(args.output_dir, safe_serialization=True)
        tokenizer.save_pretrained(args.output_dir)
        print(f"Сохранена полная модель и токенизатор в {args.output_dir}")

    print("Готово.")


if __name__ == '__main__':
    main()
