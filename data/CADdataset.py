import os
import time
import torch
import numpy as np
import json
import pdb
import cv2
import copy
import random
import torch
from glob import escape, glob
from PIL import Image
import torchvision.transforms as T

from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from pointnet_util import *

DEBUG = False


class CADDataLoader(Dataset):
    def __init__(self, root, clus_ratio=1/32, split='training', uniform=False, do_norm=True, do_clus=False, cfg=None):
        self.set_random_seed(123)
        self.root = root
        self.split = split
        self.do_clus = do_clus
        if cfg is not None:
            self.clus_num_per_batch = cfg.clus_num_per_batch
            self.nn = cfg.nn
            self.size = cfg.img_size
        else:
            self.clus_num_per_batch = 16
            self.nn = 64
            self.size = 700
        ##################### transformations ############################
        transform = [T.ToTensor()]
        if do_norm:
            transform.append(imagenet_preprocess())
        self.transform = T.Compose(transform)
        ##################### pre-loading ############################
        self.image_path_list = glob(os.path.join(self.root, "images", split, "images", "*.png"))
        self.anno_path_list = glob(os.path.join(self.root, "annotations", split, "constructed_graphs_withdeg", "*.npy"))
        self.image_path_list = sorted(self.image_path_list)
        self.anno_path_list = sorted(self.anno_path_list)
        ##################### process a special case ############################
        for img_path in self.image_path_list:
            if "0104-0102" in img_path:
                self.image_path_list.remove(img_path)

        assert len(self.image_path_list) == len(self.anno_path_list)
        self.length = len(self.image_path_list)

        print("======================= self.do_clus:", self.do_clus, "=======================")
        if self.do_clus:
            print("---> before filter_smallset:", len(self.anno_path_list))
            if not DEBUG:
                self.image_path_list, self.anno_path_list = self.filter_smallset()
                # if split == "training":
                #     self.image_path_list, self.anno_path_list = self.image_path_list[:128], self.anno_path_list[:128]
            print("---> after filter_smallset:", len(self.anno_path_list))
            self.length = len(self.image_path_list)
            # self.image_path_list = np.array_split(self.image_path_list, self.length//self.clus_num_per_batch)
            # self.anno_path_list = np.array_split(self.anno_path_list, self.length//self.clus_num_per_batch)
            # self.image_path_list = [_.tolist() for _ in self.image_path_list]
            # self.anno_path_list = [_.tolist() for _ in self.anno_path_list]
        else:
            self.length = len(self.image_path_list)

        print('The size of %s data is %d'%(self.split, self.length))

    def filter_smallset(self):
        anno_path_list_new = list()
        image_path_list_new = list()
        for idx, ann_path in enumerate(self.anno_path_list):
            adj_node_classes = np.load(ann_path, \
                                    allow_pickle=True).item()
            target = adj_node_classes["class"]
            if len(target) >= self.nn:
                anno_path_list_new.append(self.anno_path_list[idx])
                image_path_list_new.append(self.image_path_list[idx])
        return image_path_list_new, anno_path_list_new

    def __len__(self):
        return self.length

    def _get_item(self, index):
        
        img_path = self.image_path_list[index]
        ann_path = self.anno_path_list[index]
        assert os.path.basename(img_path).split(".")[0] == \
            os.path.basename(ann_path).split(".")[0] 
        
        image = Image.open(img_path).convert("RGB")
        image = image.resize((self.size, self.size))
        image = self.transform(image).cuda()

        adj_node_classes = np.load(ann_path, \
                            allow_pickle=True).item()
        target = adj_node_classes["class"]
        target = torch.from_numpy(np.array(target, dtype=np.long)).cuda()

        center = adj_node_classes["centers_normed"]
        point_set = torch.from_numpy(np.array(center, dtype=np.float32)).cuda()

        geo_feat = adj_node_classes["node"]
        geo_feat = torch.from_numpy(np.array(geo_feat, dtype=np.long)).cuda()
        
        degree = adj_node_classes["degrees"]
        degree = torch.from_numpy(np.array(degree, dtype=np.long)).cuda()
        degree = torch.clamp(degree, 0, 128, out=None).unsqueeze(-1)
        
        basename = os.path.basename(img_path)

        if self.do_clus:
            point_set_cp = copy.deepcopy(point_set)
            target_cp = copy.deepcopy(target)
            if self.split == "training": 
                point_set, target, geo_feat, degree, indexes = self.sample_and_group(\
                        self.clus_num_per_batch, self.nn, point_set.unsqueeze(0), \
                        target.unsqueeze(0), geo_feat.unsqueeze(0), degree.unsqueeze(0), rand_prob=0.2)
                point_set = point_set.squeeze(0) #Shape is:  [self.clus_num_per_batch, self.nn, 2]
                geo_feat = geo_feat.squeeze(0) #Shape is:  [self.clus_num_per_batch, self.nn, 2]
                target = target.squeeze(0)
                # self.plot_indexes(point_set_cp, indexes, basename, save_dir="/home/zhiwen/projects/Point-Transformers/data/tmp3")
            else:
                '''
                F1 drop from 0.75 to 0.72
                '''
                # if point_set.shape[0] <= 5000:
                #     max_line = point_set.shape[0] // 4
                # elif 5000 <= point_set.shape[0] <= 10000 :
                #     max_line = point_set.shape[0] // 8
                # elif 10000 <= point_set.shape[0] <= 20000 :
                #     max_line = point_set.shape[0] // 16
                # elif 20000 <= point_set.shape[0] <= 40000 :
                #     max_line = point_set.shape[0] // 32
                # else:
                #     max_line = point_set.shape[0] // 64
                # point_set, target, geo_feat, indexes = self.sample_and_group(\
                #     max_line, self.nn, point_set.unsqueeze(0), target.unsqueeze(0), geo_feat.unsqueeze(0))
                if 0 < point_set.shape[0] <= 1000:
                    div = int(4 * 2)
                elif 1000 <= point_set.shape[0] <= 5000:
                    div = int(8* 2)
                elif 5000 <= point_set.shape[0] <= 20000:
                    div = int(24 * 2)
                else:
                    div = int(48 * 2)
                point_set, target, geo_feat, degree, indexes = self.sample_and_group(\
                    point_set.shape[0]//div, self.nn, point_set.unsqueeze(0), target.unsqueeze(0), geo_feat.unsqueeze(0), degree.unsqueeze(0))
                point_set = point_set.squeeze(0) #Shape is:  [self.clus_num_per_batch, self.nn, 2]
                geo_feat = geo_feat.squeeze(0) #Shape is:  [self.clus_num_per_batch, self.nn, 2]
                target = target_cp
                # self.plot_indexes(point_set_cp, indexes, basename, save_dir="/home/zhiwen/projects/Point-Transformers/data/tmp3")
            '''
            TO REMOVE
            '''
            # point_set, target, indexes = self.sample_and_group(\
            #     self.clus_num_per_batch, self.nn, point_set.unsqueeze(0), target.unsqueeze(0))
            # point_set = point_set.squeeze(0) #Shape is:  [self.clus_num_per_batch, self.nn, 2]
            # target = target.squeeze(0)
            # indexes = torch.Tensor([1]).cuda()
        else:
            indexes = torch.Tensor([1]).cuda()
            point_set = point_set[:10000]
            target = target[:10000]

        return image, point_set, target, geo_feat, degree, indexes, basename

    def __getitem__(self, index):
        return self._get_item(index)

    def sample_and_group(self, npoint, nsample, xyz, target, geo_feat, degreee, rand_prob=0):
        '''
        Points.shape: [16, 1024, 64]
        xyz.shape: [16, 1024, 3]
        '''
        B, N, C = xyz.shape
        S = npoint 
        
        if rand_prob <= 0.001:
            fps_idx = farthest_point_sample(xyz, npoint) # [16, 1024, 3] + 512 -> [16, 512]
        else:
            if random.uniform(0, 1) < rand_prob:
                fps_idx = random_point_sample(xyz, npoint)
            else:
                fps_idx = farthest_point_sample(xyz, npoint) # [16, 1024, 3] + 512 -> [16, 512]

        new_xyz = index_points(xyz, fps_idx)  # [16, 1024, 3] + [16, 512] -> [16, 512, 3]

        dists = square_distance(new_xyz, xyz)  # [16, 512, 3] [16, 1024, 3] -> [16, 512, 1024]
        idx = dists.argsort()[:, :, :nsample]  # [16, 512, 32]

        grouped_xyz = index_points(xyz, idx)  # [16, 1024, 3] + [16, 512] -> [16, 512, 3]
        grouped_target = index_points(target, idx) # [16, 512, 32, 64]
        new_geo_feat = index_points(geo_feat, idx) # [16, 512, 32, 64]
        new_degreee = index_points(degreee, idx) # [16, 512, 32, 64]
        return grouped_xyz, grouped_target, new_geo_feat, new_degreee, idx.squeeze(0)
        
    def draw_pts(self, point_set, save_path, re_norm=True):
        point_set = point_set.cpu().numpy()
        pts_length = point_set.shape[0]
        img = np.zeros((700, 700))
        for pts in point_set:
            if re_norm:
                pts = pts*350 + 350
            pts = [int(p) for p in pts]
            cv2.circle(img, pts, 1, 255)
        cv2.imwrite(save_path, img)

    def plot_indexes(self, point_set, indexes, basename, save_dir, re_norm=True):
        os.makedirs(save_dir, exist_ok=True)
        print("===> plot_indexes")
        print("===> basename:", basename)
        color_list = PALLTE.tolist()
        color_total = len(color_list)
        point_set = point_set.cpu().numpy()
        indexes = indexes.cpu().numpy()
        img = np.zeros((700, 700, 3))
        for idx_center in range(indexes.shape[0]):
            draw = True
            color_idx = random.randint(0, color_total-1)
            color = color_list[color_idx]
            for idx_nn in range(indexes.shape[1]):
                idx = indexes[idx_center][idx_nn]
                pts = point_set[idx]
                if re_norm:
                    pts = pts*350 + 350
                pts = [int(p) for p in pts]
                if draw:
                    cv2.circle(img, pts, 5, color, -1)
                    draw = False
                else:
                    cv2.circle(img, pts, 2, color)
        cv2.imwrite(os.path.join(save_dir, basename.replace(".svg", ".png")), img)

    def set_random_seed(self, seed, deterministic=False, use_rank_shift=False):
        """Set random seed.

        Args:
            seed (int): Seed to be used.
            deterministic (bool): Whether to set the deterministic option for
                CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
                to True and `torch.backends.cudnn.benchmark` to False.
                Default: False.
            rank_shift (bool): Whether to add rank number to the random seed to
                have different random seed in different threads. Default: False.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def farthest_point_sample(self, xyz, target, npoint):
        """
        Input:
            xyz: pointcloud data, [B, N, 3]
            npoint: number of samples
        Return:
            centroids: sampled pointcloud index, [B, npoint]
        """
        if len(xyz.shape) == 2:
            xyz = xyz.unsqueeze(0)
        if len(target.shape) == 2:
            target = target.unsqueeze(0)
        device = xyz.device
        B, N, C = xyz.shape
        centroids = torch.zeros(B, npoint, dtype=torch.long).to(device) # farest index
        distance = torch.ones(B, N).to(device) * 1e10
        farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device) # len = batch
        batch_indices = torch.arange(B, dtype=torch.long).to(device)
        for i in range(npoint):
            centroids[:, i] = farthest
            centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
            dist = torch.sum((xyz - centroid) ** 2, -1) #[1, 10000]
            distance = torch.min(distance, dist) #[1, 10000] + [1, 10000] -> [1, 10000]
            farthest = torch.max(distance, -1)[1] # a number
        return xyz[0, centroids, :].squeeze(0), target[0, centroids, :].squeeze(0)


if __name__ == '__main__':
    data = CADDataLoader('data/cad', split='training', do_norm=True, do_clus=True)
    DataLoader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False)
    anno_list = AnnoList().anno_list_all_reverse
    class_num = len(anno_list)
    cnt_prd, cnt_gt, cnt_tp = [torch.Tensor([0 for x in range(0, class_num+1)]).cuda() for _ in range(3)]

    for i, (image, point_set, target, geo_feat, degree, indexes, basename) in enumerate(DataLoader):
        if i >= 100:
            break
    #     target = target.view(-1, 1)[:, 0]
    #     res_dict = dict()
    #     for ii, idx in enumerate(indexes):
    #         if idx not in res_dict.keys():
    #             res_dict[idx] = list()
    #             res_dict[idx].append(int(target[ii].cpu().numpy()))
    #         else:
    #             res_dict[idx].append(int(target[ii].cpu().numpy()))

    #     for key, val in res_dict.items():
    #         res_dict[key] = max(res_dict[key],key=res_dict[key].count)

    #     res_dict = dict(sorted(res_dict.items()))

    #     if len(list(res_dict.keys())) != target.shape[0]:
    #         res_dict_pad = dict()
    #         for iii in range(target.shape[0]):
    #             if iii in res_dict.keys():
    #                 res_dict_pad[iii] = res_dict[iii]
    #             else:
    #                 res_dict_pad[iii] = [0]
    
    #     res_tensor = list()
    #     for key, val in res_dict.items():
    #         res_tensor.append(val)
    #     res_tensor = torch.Tensor(res_tensor).to(target.device)
    
    #     for prd, gt in zip(target, target):
    #         cnt_prd[prd] += 1
    #         cnt_gt[gt] += 1
    #         if prd == gt:
    #             cnt_tp[gt] += 1

    # for cls_id in range(1, class_num):
    #     precision = cnt_tp[cls_id]/(cnt_prd[cls_id]+1e-4)
    #     recall = cnt_tp[cls_id]/(cnt_gt[cls_id]+1e-4)
    #     f1 = (2*precision*recall) / (precision + recall+1e-4)
    #     print("ID:[{:2s}], CLASS:[{:15s}], Pred Num: [{:0>7}], GT Num: [{:0>7}], F1:[{:.2%}], Precision:[{:.2%}], Recall:[{:.2%}]".format(\
    #         str(cls_id), anno_list[cls_id], cnt_prd[cls_id], cnt_gt[cls_id], f1, precision, recall))
    #     # print(point.shape)
    #     # print(label.shape)