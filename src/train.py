import argparse
import matplotlib.pyplot as plt
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def seed_everything(seed=2003):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def create_dataloader():
    pass

def calculate_DPO_loss():
    pass

def train():
    pass

def main():
    pass

if __name__ == "__main__":
    main()
