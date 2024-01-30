import argparse
import random
import numpy as np

import torch

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments

from trl import DPOTrainer

import wandb

def seed_everything(seed=2003):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def preprocess_data(item):
    return {
        'prompt': 'Instruct: ' + item['prompt'] + '\n',
        'chosen': 'Output: ' + item['chosen'],
        'rejected': 'Output: ' + item['rejected']
    }

def train(model, ref_model, dataset, tokenizer, beta, training_args):
    model.train()
    ref_model.eval()

    dpo_trainer = DPOTrainer(
        model,
        ref_model,
        beta=beta,
        train_dataset=dataset,
        tokenizer=tokenizer,
        args=training_args,
        max_length=1024,
        max_prompt_length=512
    )

    dpo_trainer.train()

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=2003)
    parser.add_argument("--model_name", type=str, default="microsoft/phi-2")
    parser.add_argument("--dataset_name", type=str, default="jondurbin/truthy-dpo-v0.1")
    parser.add_argument("--wandb_project", type=str, default="truthy-dpo")

    args = parser.parse_args()

    seed_everything(args.seed)

    wandb.login()
    wandb.init(project=args.wandb_project, config=args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
    ref_model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)

    dataset = load_dataset(args.dataset_name, split="train")
    dataset = dataset.map(preprocess_data)

    training_args = TrainingArguments(
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        report_to="wandb",
        output_dir='./results',
        logging_steps=10,
        remove_unused_columns=False,
    )

    train(model, ref_model, dataset, tokenizer, args.beta, training_args)

    model.save_pretrained("model-HF-DPO.pt")

if __name__ == "__main__":
    main()
