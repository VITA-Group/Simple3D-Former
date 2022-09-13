# ImageeNet Dataloader

import os
import sys
import numpy as np
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
import torch.nn.functional as F

if __name__ == '__main__':
    P = torch.rand(4,4,3)
    Q = torch.rand(4,4,3)

    print((P * (P / Q).log()).mean(1))
    # tensor(0.0863), 10.2 µs ± 508

    print(F.kl_div(Q.log(), P, reduction='batchmean'))