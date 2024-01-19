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

def calculate_DPO_loss():
    pass

def train():
    pass

def main():
    pass

if __name__ == "__main__":
    main()
