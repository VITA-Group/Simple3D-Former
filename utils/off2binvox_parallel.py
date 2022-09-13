#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
File: off2binvox.py
Created: 2020-01-21 21:32:40
Author : Yangmaonan
Email : 59786677@qq.com
Description: 将 ModelNet10 数据集中.off文件转为binvox文件
'''
from multiprocessing import Pool
import os
import glob

DATA_ROOT = '../data/ModelNet10'
MODELNET_40_DATA_ROOT = '../../../data/ModelNet40_Aligned'

CLASSES = {'bathtub', 'chair', 'dresser', 'night_stand', 'sofa', 'toilet', 'bed', 'desk', 'monitor', 'table'}
CLASSES_40 = {'airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair','cone',
              'cup','curtain','desk','door','dresser','flower_pot','glass_box','guitar','keyboard','lamp',
              'laptop','mantel','monitor','night_stand','person', 'piano','plant','radio','range_hood','sink',
              'sofa','stairs','stool','table','tent','toilet','tv_stand','vase','wardrobe','xbox'
        }

def converter(off):
    os.system(f'./binvox -d 32 -cb -pb {off}')

DATA_ROOT=MODELNET_40_DATA_ROOT
CLASSES=CLASSES_40

filename = []
for c in CLASSES:
    for split in ['test', 'train']:
        for off in glob.glob(os.path.join(DATA_ROOT, c, split, '*.off')):
            # 判断是否存在
            binname = os.path.join(DATA_ROOT, c, split, os.path.basename(off).split('.')[0] + '.binvox')
            if os.path.exists(binname):
                print(binname, "exits, continue...", os.stat(binname).st_size)
                continue
            #os.system(f'./binvox -d 32 -cb -pb {off}')
            filename.append(off)

print(len(filename))

with Pool(16) as p:
    p.map(converter,filename)
