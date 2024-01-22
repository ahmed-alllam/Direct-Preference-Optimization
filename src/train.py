import argparse
import random
import numpy as np
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

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

    reward_accuracies = (prefered_relative_logprob > disprefered_relative_logprob).float().mean(dim=-1)
    reward_margins = (prefered_relative_logprob - disprefered_relative_logprob).mean(dim=-1)

    loss = -F.logsigmoid(beta * (prefered_relative_logprob - disprefered_relative_logprob)).mean(dim=-1)

    return loss, prefered_relative_logprob.mean(dim=-1), disprefered_relative_logprob.mean(dim=-1), reward_accuracies, reward_margins

def get_log_prob(logits, labels):
    log_probs = F.log_softmax(logits, dim=-1)
    return torch.gather(log_probs, -1, labels.unsqueeze(-1)).squeeze(-1).mean(-1)

def collate_fn(batch, tokenizer, max_length, device):
    prompts = ['Instruct: ' + item['prompt'] + '\n' for item in batch]
    chosen_responses = ['Output: ' + item['chosen'] for item in batch]
    rejected_responses = ['Output: ' + item['rejected'] for item in batch]

    prompt_ids = tokenizer.batch_encode_plus(prompts, padding=True, return_tensors="pt", max_length=max_length, truncation=True)['input_ids'].to(device)
    prefered_ids = tokenizer.batch_encode_plus(chosen_responses, padding=True, return_tensors="pt", max_length=max_length, truncation=True)['input_ids'].to(device)
    disprefered_ids = tokenizer.batch_encode_plus(rejected_responses, padding=True, return_tensors="pt", max_length=max_length, truncation=True)['input_ids'].to(device)

    prompt_prefered_ids = torch.cat([prompt_ids, prefered_ids], dim=-1)
    prompt_disprefered_ids = torch.cat([prompt_ids, disprefered_ids], dim=-1)

    prompt_prefered_mask = torch.cat([torch.ones_like(prompt_ids), torch.zeros_like(prefered_ids)], dim=-1)
    prompt_disprefered_mask = torch.cat([torch.ones_like(prompt_ids), torch.zeros_like(disprefered_ids)], dim=-1)

    return {'prompt_prefered_ids': prompt_prefered_ids,
            'prompt_disprefered_ids': prompt_disprefered_ids,
            'prompt_prefered_mask': prompt_prefered_mask,
            'prompt_disprefered_mask': prompt_disprefered_mask}

def train(model, ref_model, tokenizer, optimizer, train_dataloader, epochs=1, beta=0.1):
    model.train()
    ref_model.eval()

    for epoch in range(epochs):
        for batch in tqdm(train_dataloader):
            optimizer.zero_grad()

            prompt_prefered_ids = batch['prompt_prefered_ids']
            prompt_disprefered_ids = batch['prompt_disprefered_ids']
            prompt_prefered_mask = batch['prompt_prefered_mask']
            prompt_disprefered_mask = batch['prompt_disprefered_mask']

            model_prefered_log_prob = get_log_prob(model(prompt_prefered_ids, attention_mask=prompt_prefered_mask).logits, prompt_prefered_ids)
            model_disprefered_log_prob = get_log_prob(model(prompt_disprefered_ids, attention_mask=prompt_disprefered_mask).logits, prompt_disprefered_ids)

            ref_prefered_log_prob = get_log_prob(ref_model(prompt_prefered_ids, attention_mask=prompt_prefered_mask).logits, prompt_prefered_ids)
            ref_disprefered_log_prob = get_log_prob(ref_model(prompt_disprefered_ids, attention_mask=prompt_disprefered_mask).logits, prompt_disprefered_ids)

            loss, prefered_relative_logprob, disprefered_relative_logprob, reward_accuracies, reward_margins = calculate_DPO_loss(model_prefered_log_prob, model_disprefered_log_prob,
                                          ref_prefered_log_prob, ref_disprefered_log_prob,
                                          beta=beta)

            loss.backward()
            optimizer.step()

            wandb.log({'loss': loss.item(),
                       'prefered_relative_logprob': prefered_relative_logprob,
                       'disprefered_relative_logprob': disprefered_relative_logprob,
                       'reward_accuracy': reward_accuracies,
                       'reward_margin': reward_margins})

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=2003)
    parser.add_argument("--model_name", type=str, default="microsoft/phi-2")
    parser.add_argument("--dataset_name", type=str, default="jondurbin/truthy-dpo-v0.1")
    parser.add_argument("--wandb-project", type=str, default="math-preference-dpo")

    args = parser.parse_args()

    seed_everything(args.seed)

    wandb.login()
    wandb.init(project=args.wandb_project, config=args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
    ref_model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr)

    dataset = load_dataset(args.dataset_name, split="train")
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=partial(collate_fn, tokenizer=tokenizer, max_length=args.max_length, device=device))

    train(model, ref_model, tokenizer, optimizer, train_dataloader, epochs=args.epochs, beta=args.beta)

    model.save_pretrained("model-DPO.pt")

if __name__ == "__main__":
    main()
