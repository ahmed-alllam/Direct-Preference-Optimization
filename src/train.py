import argparse
import matplotlib.pyplot as plt
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel

from tqdm import tqdm
import wandb

from utils import seed_everything, create_dataloader

def calculate_DPO_losses(model_prefered_logits, model_disprefered_logits,
                       ref_prefered_logits, ref_disprefered_logits,
                       beta=0.5):

    relative_prefered_logits = model_prefered_logits - ref_prefered_logits
    relative_disprefered_logits = model_disprefered_logits - ref_disprefered_logits

    return -F.logsigmoid(beta * relative_prefered_logits - beta * relative_disprefered_logits)

def train():
    pass

def main():
    pass

if __name__ == "__main__":
    main()
