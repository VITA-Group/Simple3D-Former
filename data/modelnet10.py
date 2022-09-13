# coding: utf-8
import os
import sys
import numpy as np
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
sys.path.insert(0, '../utils/')
sys.path.insert(0, './utils/')
import binvox_rw


class ModelNet10(Dataset):
    def __init__(self, data_root, n_classes, idx2cls, split='train'):
        """
        Args:
            split (str, optional): 'train' or 'test'. Defaults to 'train'.
        """
        self.data_root = data_root
        self.n_classes = n_classes
        self.samples_str = []
        self.cls2idx = {}
        for k, v in idx2cls.items():
            self.cls2idx.update({v: k})
            for sample_str in glob.glob(os.path.join(data_root, v, split, '*.binvox')):
                if re.match(r"[a-zA-Z_]+_\d+.binvox", os.path.basename(sample_str)):
                    self.samples_str.append(sample_str)
        print(self.cls2idx)

    def __getitem__(self, idx):
        sample_name = self.samples_str[idx]
        cls_name = re.split(r"_\d+\.binvox", os.path.basename(sample_name))[0]
        cls_idx = self.cls2idx[cls_name]
        with open(sample_name, 'rb') as file:
            data = np.int32(binvox_rw.read_as_3d_array(file).data)
            data = data[np.newaxis, :]

        sample = {'voxel': data, 'cls_idx': cls_idx}

        return sample

    def __len__(self):
        return len(self.samples_str)


if __name__ == "__main__":
    idx2cls = {0: 'bathtub', 1: 'chair', 2: 'dresser', 3: 'night_stand',
               4: 'sofa', 5: 'toilet', 6: 'bed', 7: 'desk', 8: 'monitor', 9: 'table'}

    data_root = './ModelNet10'

    dataset = ModelNet10(data_root=data_root, n_classes=10, idx2cls=idx2cls, split='train')
    cnt = len(dataset)

    data, cls_idx = dataset[0]['voxel'], dataset[1]['cls_idx']
    print(f"length: {cnt}\nsample data: {data}\nsample cls: {cls_idx}")
