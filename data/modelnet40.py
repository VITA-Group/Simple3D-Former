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
import binvox_rw, data_augmentation




class ModelNet40(Dataset):
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
        print("There are {} samples in the dataset.".format(len(self.samples_str)))


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
    
    def class_weight(self):
        cls_name = [name.split('/')[-4] for name in self.samples_str]
        cls_idx = torch.LongTensor([self.cls2idx[name] for name in cls_name])
        class_freq= torch.bincount(cls_idx)
        class_weight = 1.0/torch.log1p(1.+ class_freq)
        class_weight = len(class_weight) * class_weight/ torch.sum(class_weight)
        print(class_weight)
        return class_weight


class ModelNet40_Constrastive(ModelNet40):
    def __init__(self, data_root, n_classes, idx2cls, split='train'):
        """
        Args:
            split (str, optional): 'train' or 'test'. Defaults to 'train'.
        """
        super().__init__(data_root, n_classes, idx2cls, split=split)

    def __getitem__(self, idx):
        sample_name = self.samples_str[idx]
        cls_name = re.split(r"_\d+\.binvox", os.path.basename(sample_name))[0]
        cls_idx = self.cls2idx[cls_name]
        with open(sample_name, 'rb') as file:
            data = np.int32(binvox_rw.read_as_3d_array(file).data)
            data = data[np.newaxis, :]
        try:
            with open(sample_name, 'rb') as file:
                contrastive = np.int32(data_augmentation.add_affine_transformation_to_voxel(file).data)
                contrastive = contrastive[np.newaxis, :]
        except:
            #print(sample_name)
            contrastive = data

        sample = {'voxel': data, 'cls_idx': cls_idx, 'contrastive': contrastive}

        return sample

    def __len__(self):
        return len(self.samples_str)




if __name__ == "__main__":
    idx2cls = {
        0:'airplane', 1:'bathtub', 2:'bed', 3:'bench',
        4:'bookshelf', 5:'bottle', 6:'bowl', 7:'car',
        8:'chair', 9:'cone', 10:'cup', 11:'curtain',
        12:'desk', 13:'door', 14:'dresser', 15:'flower_root',
        16:'glass_box', 17:'guitar', 18:'keyboard', 19:'lamp',
        20:'laptop', 21:'mantel', 22:'monitor', 23:'night_stand',
        24:'person', 25:'piano', 26:'plant', 27:'radio',
        28:'range_hood', 29:'sink', 30:'sofa', 31:'stairs',
        32:'stool', 33:'table', 34:'tent', 35:'toilet',
        36:'tv_stand', 37:'vase', 38:'wardrobe', 39:'xbox'
    }

    data_root = '/mnt/storage/yiwang/data/ModelNet40_Aligned'

    dataset = ModelNet40_Constrastive(data_root=data_root, n_classes=40, idx2cls=idx2cls, split='train')
    cnt = len(dataset)

    data, cls_idx = dataset[0]['voxel'], dataset[1]['cls_idx']
    print(f"length: {cnt}\nsample data: {np.count_nonzero(data)}\nsample cls: {cls_idx}")
