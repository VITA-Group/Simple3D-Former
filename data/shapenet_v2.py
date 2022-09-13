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
import tqdm


class ShapeNetV2(Dataset):
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
            for sample_str in glob.glob(os.path.join(data_root, v, '*/models/*.solid.binvox')):
                if re.match(r"[a-z_A-Z]+.solid+.binvox", os.path.basename(sample_str)):
                    self.samples_str.append(sample_str)
                    
        print("There are {} samples in the dataset.".format(len(self.samples_str)))


    def __getitem__(self, idx):
        sample_name = self.samples_str[idx]
        model_id = sample_name.split('/')[-3]
        cls_name = sample_name.split('/')[-4]
        cls_idx = self.cls2idx[cls_name]
        with open(sample_name, 'rb') as file:
            data = np.int32(binvox_rw.read_as_3d_array(file).data)
            data = data[np.newaxis, :]

        sample = {'voxel': data, 'cls_idx': cls_idx, 'model_id': model_id} 
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

class ShapeNetV2_Contrastive(ShapeNetV2):
    def __init__(self, data_root, n_classes, idx2cls, split='train'):
        """
        Args:
            split (str, optional): 'train' or 'test'. Defaults to 'train'.
        """
        super().__init__(data_root, n_classes, idx2cls, split=split)
        m = torch.nn.MaxPool3d(4)
        print("Start create a contrastive dataset!\n")
        count = 0
        for sample_name in tqdm.tqdm(self.samples_str):
            if os.path.exists(sample_name+'.npy'):
               continue
            with open(sample_name, 'rb') as file:
               try:
                   new_voxel = data_augmentation.add_affine_transformation_to_voxel(file)
                   count = count + 1
               except:
                   file.seek(0,0)
                   new_voxel = binvox_rw.read_as_3d_array(file)
               with open(sample_name+'.npy', 'wb') as outfile:
                   data = m(torch.from_numpy(new_voxel.data[np.newaxis,np.newaxis, :]).float()).int().numpy()
                   data = np.squeeze(data)
                   np.save(outfile, data)
        print("Complete contrastive dataset, %d models has been created!\n" % count)
        


    def __getitem__(self, idx):
        sample_name = self.samples_str[idx]
        cls_name = sample_name.split('/')[-4]
        cls_idx = self.cls2idx[cls_name]
        
        assert os.path.exists(sample_name+'.npy')

        with open(sample_name, 'rb') as file:
            data = np.int32(binvox_rw.read_as_3d_array(file).data)
            data = data[np.newaxis, :]
        with open(sample_name+'.npy', 'rb') as file:
            contrastive = np.load(file)
            contrastive = contrastive[np.newaxis, :]
      
        sample = {'voxel': data, 'cls_idx': cls_idx, 'contrastive': contrastive}

        return sample

    def __len__(self):
        return len(self.samples_str)


if __name__ == "__main__":
    idx2cls = {0: '02691156', 1: '02747177', 2: '02773838', 3: '02801938',
               4: '02808440', 5: '02818832', 6: '02828884', 7: '02843684', 
               8: '02871439', 9: '02876657',10: '02880940',11: '02924116',
               12:'02933112',13: '02942699',14: '02946921',15: '02954340',
               16:'02958343',17: '02992529',18: '03001627',19: '03046257',
               20:'03085013',21: '03207941',22: '03211117',23: '03261776',
               24:'03325088',25: '03337140',26: '03467517',27: '03513137',
               28:'03593526',29: '03624134',30: '03636649',31: '03642806',
               32:'03691459',33: '03710193',34: '03759954',35: '03761084',
               36:'03790512',37:'03797390',38:'03928116',39:'03938244',
               40:'03948459',41:'03991062',42:'04004475',43:'04074963',
               44:'04090263',45:'04099429',46:'04225987',47:'04256520',
               48:'04330267',49:'04379243',50:'04401088',51:'04460130',
               52:'04468005',53:'04530566',54:'04554684'
               }

    data_root = '/mnt/storage/datasets/ShapeNetCore_v2'

    dataset = ShapeNetV2(data_root=data_root, n_classes=55, idx2cls=idx2cls, split='train')
    cnt = len(dataset)

    data, cls_idx = dataset[0]['voxel'], dataset[1]['cls_idx']
    print(f"length: {cnt}\nsample data: {data}\nsample cls: {cls_idx}")
    print(cnt)
