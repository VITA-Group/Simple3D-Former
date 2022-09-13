#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
File: off2binvox.py
Created: 2020-01-21 21:32:40
Author : Yangmaonan
Email : 59786677@qq.com
Description: 将 ModelNet10 数据集中.off文件转为binvox文件
'''
# TODO: 使用多进程加速文件转换速度
import os
import glob

DATA_ROOT = '../data/ModelNet10'

CLASSES = {'bathtub', 'chair', 'dresser', 'night_stand', 'sofa', 'toilet', 'bed', 'desk', 'monitor', 'table'}

for c in CLASSES:
    for split in ['test', 'train']:
        for off in glob.glob(os.path.join(DATA_ROOT, c, split, '*.off')):
            # 判断是否存在
            binname = os.path.join(DATA_ROOT, c, split, os.path.basename(off).split('.')[0] + '.binvox')
            if os.path.exists(binname):
                print(binname, "exits, continue...")
                continue
            os.system(f'./binvox -d 32 -cb -pb {off}')
