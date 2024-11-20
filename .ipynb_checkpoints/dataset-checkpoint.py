import numpy as np 
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import datasets

dataset = datasets.load_dataset("ccdv/arxiv-summarization", split='train', streaming=True)
raw_dataset = list(dataset.take(3500))

print(raw_dataset[0])
