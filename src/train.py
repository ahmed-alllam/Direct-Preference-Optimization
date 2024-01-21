import argparse
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as AdamW

from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel

import wandb
from tqdm import tqdm

def seed_everything(seed=2003):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def calculate_DPO_loss(model_prefered_logprob, model_disprefered_logprob,
                       ref_prefered_logprob, ref_disprefered_logprob,
                       beta=0.5):

    prefered_relative_logprob = model_prefered_logprob - ref_prefered_logprob
    disprefered_relative_logprob = model_disprefered_logprob - ref_disprefered_logprob

    return -torch.mean(F.logsigmoid(beta * prefered_relative_logprob - beta * disprefered_relative_logprob))

def get_log_prob(logits, labels):
    log_probs = F.log_softmax(logits, dim=-1)
    return torch.gather(log_probs, -1, labels.unsqueeze(-1)).squeeze(-1).sum(-1)

def train(model, ref_model, optimizer, train_dataloader, epochs=1, beta=0.5):
    model.train()
    ref_model.eval()

    for epoch in range(epochs):
        for batch in tqdm(train_dataloader):
            optimizer.zero_grad()
            prompt_ids = batch['prompt_ids'].to(device)
            prefered_ids = batch['prefered_ids'].to(device)
            disprefered_ids = batch['disprefered_ids'].to(device)

            model_prefered_log_prob = get_log_prob(model(prompt_ids).logits, prefered_ids)
            model_disprefered_log_prob = get_log_prob(model(prompt_ids).logits, disprefered_ids)

            ref_prefered_log_prob = get_log_prob(ref_model(prompt_ids).logits, prefered_ids)
            ref_disprefered_log_prob = get_log_prob(ref_model(prompt_ids).logits, disprefered_ids)

            loss = calculate_DPO_loss(model_prefered_logits, model_disprefered_logits,
                                      ref_prefered_logits, ref_disprefered_logits,
                                      beta=beta)

            loss.backward()
            optimizer.step()

            wandb.log({"loss": loss.item(), "epoch": epoch})

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=2003)
    parser.add_argument("--model_name", type=str, default="microsoft/phi-2")
    parser.add_argument("--dataset_name", type=str, default="argilla/distilabel-math-preference-dpo")
    parser.add_argument("--wandb-project", type=str, default="math-preference-dpo")

    args = parser.parse_args()
    tokenizer.pad_token = tokenizer.eos_token

    seed_everything(args.seed)

    wandb.init(project=args.wandb_project, config=args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
    ref_model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr)

    dataset = load_dataset(args.dataset_name, split="train")

    def collate_fn(batch):
        prompts = [item['prompt'] for item in batch]
        chosen_responses = [item['chosen_response'] for item in batch]
        rejected_responses = [item['rejected_response'] for item in batch]

        prompt_ids = tokenizer.batch_encode_plus(prompts, padding=True, return_tensors="pt")['input_ids'].to(device)
        prefered_ids = tokenizer.batch_encode_plus(chosen_responses, padding=True, return_tensors="pt")['input_ids'].to(device)
        disprefered_ids = tokenizer.batch_encode_plus(rejected_responses, padding=True, return_tensors="pt")['input_ids'].to(device)

        return {"prompt_ids": prompt_ids, "prefered_ids": prefered_ids, "disprefered_ids": disprefered_ids}

    train_dataloader = torch.utils.data.DataLoader(dataset['train'], batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    train(model, ref_model, optimizer, train_dataloader, epochs=args.epochs, beta=args.beta)

if __name__ == "__main__":
    main()
