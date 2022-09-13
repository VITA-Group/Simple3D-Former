#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
Adapted from https://github.com/MonteYang/VoxNet.pytorch
'''

import argparse
import sys
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
from tqdm import tqdm
from models.vit_3d_2d_pretrain import *
from models.embed_layer_3d_modality import *
from models.DeIT import deit_base_patch16_224
from data.modelnet10 import ModelNet10
from data.modelnet40 import ModelNet40
from data.shapenet_v2 import ShapeNetV2
from global_var import *

import pytorch_warmup as warmup
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision.transforms as transforms
import warnings
import torchvision
from datetime import date
from datetime import timedelta
from torch.utils.data import DataLoader, Subset
import numpy as np
from ptflops import get_model_complexity_info

today = date.today()

# dd_mm_YYd
day = today.strftime("%d_%m_%Y")

torch.hub.set_dir('./cls')


VALID_EMBED_LAYER={
    'VoxelEmbed': VoxelEmbed(embed_dim=768),
    'VoxelEmbed_no_zdim': VoxelNaiveProjection(),
    'VoxelEmbed_no_average': VoxelEmbed_no_average(),
    'VoxelEmbed_14': VoxelEmbed(cell_size=9, patch_size=14),
    'VoxelEmbed_no_average_14': VoxelEmbed_no_average(cell_size=9, patch_size=14),
    'VoxelEmbed_no_zdim_14': VoxelNaiveProjection(cell_size=9, patch_size=14),
}

BACKBONE_EMBED_DIM={
    'deit_base_patch16_224': 768,
    'deit_small_patch16_224': 384,
    'deit_tiny_patch16_224': 192
}



def blue(x): return '\033[94m' + x + '\033[0m'

def find_free_port():
    import socket
    s = socket.socket()
    s.bind(('', 0))            # Bind to a free port provided by the host.
    return s.getsockname()[1]  # Return the port number assigned.

### DDP setup
def setup(rank, world_size, dist_url):
    dist.init_process_group(backend='nccl', init_method=dist_url, world_size=world_size, rank=rank,  timeout=timedelta(seconds=30))

def cleanup():
    dist.destroy_process_group()


def train(gpu, args):

    if args.slurm:
        rank = args.rank
    else:
        rank = args.rank * args.gpus + gpu
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    device = torch.device("cuda:%d" % gpu)
    torch.cuda.set_device(device)
    import socket
    ip = socket.gethostbyname(socket.gethostname())
    print(ip)
    print(f"Running basic mp on {args.dist_url}, node rank {args.rank}, gpu id {gpu}.")
    setup(rank, args.world_size, args.dist_url)
    print(f'Finish setup the process')




    if args.dataset == 'ModelNet10':
        #TODO: change this accordingly
        CLASSES= CLASSES_ModelNet10
        N_CLASSES=len(CLASSES)
        voxel_size = 30
        train_dataset = ModelNet10(data_root=args.data_root, n_classes=N_CLASSES, idx2cls=CLASSES, split='train')
        test_dataset = ModelNet10(data_root=args.data_root, n_classes=N_CLASSES, idx2cls=CLASSES, split='test')
    elif args.dataset == 'ShapeNetV2':
        CLASSES= CLASSES_SHAPENET
        N_CLASSES=len(CLASSES)
        dataset = ShapeNetV2(data_root=args.data_root, n_classes=N_CLASSES, idx2cls=CLASSES)
        print("There are %d models in ShapeNetCoreV2" % len(dataset))
        voxel_size = 128
        train_size = int(0.8*len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(9))
        VALID_EMBED_LAYER['VoxelEmbed'] = VoxelEmbed(voxel_size=128, cell_size=args.cell_size, patch_size=args.patch_size, embed_dim=BACKBONE_EMBED_DIM[args.transformer_name])
        VALID_EMBED_LAYER['VoxelEmbed_no_zdim'] = VoxelNaiveProjection(voxel_size=128, cell_size=args.cell_size, patch_size=args.patch_size, embed_dim=BACKBONE_EMBED_DIM[args.transformer_name])
        VALID_EMBED_LAYER['VoxelEmbed_no_average'] = VoxelEmbed_no_average(voxel_size=128, cell_size=args.cell_size, patch_size=args.patch_size, embed_dim=BACKBONE_EMBED_DIM[args.transformer_name])
        
    elif args.dataset == 'ModelNet40':
        CLASSES= CLASSES_ModelNet40
        N_CLASSES=len(CLASSES)
        voxel_size = 30
        train_dataset = ModelNet40(data_root=args.data_root, n_classes=N_CLASSES, idx2cls=CLASSES, split='train')
        test_dataset = ModelNet40(data_root=args.data_root, n_classes=N_CLASSES, idx2cls=CLASSES, split='test')
        VALID_EMBED_LAYER['VoxelEmbed'] = VoxelEmbed(voxel_size=30, cell_size=args.cell_size, patch_size=args.patch_size, embed_dim=BACKBONE_EMBED_DIM[args.transformer_name])
        VALID_EMBED_LAYER['VoxelEmbed_no_zdim'] = VoxelNaiveProjection(voxel_size=30, cell_size=args.cell_size, patch_size=args.patch_size, embed_dim=BACKBONE_EMBED_DIM[args.transformer_name])
        VALID_EMBED_LAYER['VoxelEmbed_no_average'] = VoxelEmbed_no_average(voxel_size=30, cell_size=args.cell_size, patch_size=args.patch_size, embed_dim=BACKBONE_EMBED_DIM[args.transformer_name])
        
    else:
        CLASSES= CLASSES_SHAPENET
        N_CLASSES=len(CLASSES)
        dataset = ShapeNetV2(data_root=args.data_root, n_classes=N_CLASSES, idx2cls=CLASSES)
        train_size = int(0.8*len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(args.manualSeed))

    
    try:
        embedding = VALID_EMBED_LAYER[args.embed_layer]
    except:
        print("Unknown type of 3D data embedding!")
        raise ValueError
    if args.reweighted:
        weights = dataset.class_weight()
    else:
        weights = None

    if args.model_name == "Voxel3D_2DPretrain":
        model = Feature3D_ViT2D_V2(embed_layer = embedding,
            n_classes=N_CLASSES, transformer_backbone=args.transformer_name,
            pretrained=args.pretrained, pos_embedding=args.pos_embedding, head=args.head).to(device)
    else:
        raise ValueError("Unknown model name!")
    if args.gpus >1:
        model = DDP(model,
        device_ids = [gpu],
        output_device= gpu,
        broadcast_buffers=False
        )
        train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=args.world_size,
        rank=rank)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batchSize, shuffle=False, num_workers=1, pin_memory=True, sampler=train_sampler)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batchSize, shuffle=True, num_workers=1, pin_memory=True)
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=args.batchSize, shuffle=True, num_workers=10)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batchSize, shuffle=True, num_workers=10)
    
    lwf= args.lwf


    if args.gpus == 1:
        teacher_data_root = './data/ImageNet/ILSVRC/Data/CLS-LOC'
        traindir = os.path.join(teacher_data_root, 'train')
        valdir = os.path.join(teacher_data_root, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        if lwf:
            teacher = deit_base_patch16_224(pretrained=True).to(device)
            teacher.eval()
            teacher_dataset = torchvision.datasets.ImageFolder(
                valdir,
                transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]))
            sample_ds = Subset(teacher_dataset, np.arange(len(train_dataloader)*args.batchSize))
            valDataLoader = torch.utils.data.DataLoader(sample_ds, batch_size=args.batchSize, shuffle=True, num_workers=10)
   

    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    #TODO: This line is fishy
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)

    if args.pretrained:
        output_folder =os.path.join(args.outf, '%s/%s/%s_%s/%s' % (day, args.model_name, args.embed_layer, args.pos_embedding, args.transformer_name))
    else:
        output_folder =os.path.join(args.outf, '%s/%s_no_pretrain/%s_%s/%s' % (day, args.model_name, args.embed_layer, args.pos_embedding, args.transformer_name))
    os.makedirs(output_folder, exist_ok=True)


    if args.model != '':
        #model.load_state_dict({k.replace('module.',''):v for k,v in torch.load(args.model, map_location=device).items()})
        model.load_state_dict(torch.load(args.model, map_location=device))
    # sanity check

    # for each in dataset:
    #     sidx = each['synset_id']
    #     midx = each['model_id']
    #     print("Reading .tmp/{synset_id}/{model_id}.png".format(synset_id=sidx, model_id=midx))
    #     a = torchvision.io.read_image('.tmp/{synset_id}/{model_id}.png'.format(synset_id=sidx, model_id=midx))
    # print("pass sanity check")
    best_acc = 0.0
    best_epoch = 0
    lambda_weight = 0.1

    print("Start training loop")

    if rank == 0:
        macs, params = get_model_complexity_info(model, (1, voxel_size, voxel_size, voxel_size), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))
        
        torch.save(model.state_dict(), '%s/epoch_0.pth' % output_folder)

    
    
    for epoch in range(args.epochs):
        # Train
        ''' adding memorizing content '''
        
        if lwf:

            for i, (sample, image) in tqdm(enumerate(zip(train_dataloader,valDataLoader), 0), total = len(train_dataloader)):
                if sample == None or image == None:
                    break
                model.train()
                voxel, cls_idx = sample['voxel'], sample['cls_idx']
                    
                voxel, cls_idx = voxel.to(device), cls_idx.to(device)
                voxel = voxel.float()
                optimizer.zero_grad()
                
                #print(rendered_image.shape)
                pred = model(voxel)

                if weights is not None:
                    loss = F.cross_entropy(pred, cls_idx, weight=weights.to(device))
                else:
                    loss = F.cross_entropy(pred, cls_idx)

                (images, labels) = image
                images, labels = images.float().to(device), labels.long().to(device)
                
                img_pred = model.forward_images(images)
                img_teacher = teacher(images)
                label_teacher = img_teacher.data.max(1)[1]
                #loss = lambda_weight * (0.5*torch.nn.functional.kl_div(img_pred.log(), img_tdeacher, reduction="mean")
                #        + 0.5*torch.nn.functional.kl_div(img_teacher.log(), img_pred, reduction="mean"))
                loss = loss + lambda_weight * F.cross_entropy(img_pred, label_teacher)
                loss.backward()
                optimizer.step()

        else: 
            for i, sample in tqdm(enumerate(train_dataloader), total = len(train_dataloader)):
                model.train()
                voxel, cls_idx = sample['voxel'], sample['cls_idx']
                    
                voxel, cls_idx = voxel.to(device), cls_idx.to(device)
                voxel = voxel.float()
                optimizer.zero_grad()
                
                #print(rendered_image.shape)
                pred = model(voxel)

                if weights is not None:
                    loss = F.cross_entropy(pred, cls_idx, weight=weights.to(device))
                else:
                    loss = F.cross_entropy(pred, cls_idx)
                
                loss.backward()
                optimizer.step()
                # pred_choice = pred.data.max(1)[1]
                # correct = pred_choice.eq(cls_idx.data).cpu().sum()
                #print('[%d: %d/%d] train loss: %f accuracy: %f' %
                #       (epoch, i, int(len(dataset)/args.batchSize), loss.item(), correct.item() / float(args.batchSize)))
        scheduler.step(scheduler.last_epoch+1)
        warmup_scheduler.dampen()
            


        if rank == 0:

            total_correct = 0
            total_testset = 0
            class_correct = torch.zeros((N_CLASSES, ))
            class_total = torch.zeros((N_CLASSES, ))
            

            # Test
            with torch.no_grad():
                for i, sample in tqdm(enumerate(test_dataloader, 0), total= len(test_dataloader)):
                    model.eval()
                    voxel, cls_idx = sample['voxel'], sample['cls_idx']
                    voxel, cls_idx = voxel.to(device), cls_idx.to(device)

                    voxel = voxel.float()
                    pred = model(voxel)
                    pred_choice = pred.data.max(1)[1]
                    correct = pred_choice.eq(cls_idx.data).cpu().sum()
                    total_correct += correct.item()
                    total_testset += voxel.shape[0]
                    pred_choice = pred_choice.cpu()
                    cls_idx = cls_idx.cpu()
                    
                    for j in range(len(pred_choice)):
                        class_correct[cls_idx[j]] += (pred_choice[j]==cls_idx[j])
                        class_total[cls_idx[j]] += 1
                
                class_acc = class_correct / class_total
  
                print("Total test samples: {}".format(total_testset))
                print("Epoch %d test accuracy %f, mean class accuracy %f" % (epoch, total_correct / float(total_testset), class_acc.sum()/ N_CLASSES))
                if total_correct / float(total_testset) >= best_acc:
                    best_acc = total_correct / float(total_testset)
                    best_epoch = epoch
                    torch.save(model.state_dict(), '%s/epoch_best.pth' % output_folder)
        

        if args.world_size > 1:
            dist.barrier()
        
    if rank == 0:
        print("Best test accuracy: epoch %d test accuracy %f" % (best_epoch, best_acc))

    cleanup()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, default='./data/ShapeNetCore_v2', help="dataset path")
    parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
    parser.add_argument('--outf', type=str, default='./cls', help='output folder')
    parser.add_argument('--model', type=str, default='', help='checkpoint model path')
    parser.add_argument('--dataset', type=str, default='ShapeNetV2', help='which dataset to be used')
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N', help='number of nodes')
    parser.add_argument('-g', '--gpus', default=1, type=int, help='number of gpus per node')
    parser.add_argument('-rank', '--rank', default=0, type=int, help='ranking within the nodes')
    parser.add_argument('--port', default='12313', type=str, metavar='P', help='port number for parallel')
    parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--model-name', type=str, default='Voxel3D_2DPretrain', help='which model to use')
    parser.add_argument('--transformer-name', type=str, default='deit_base_patch16_224', help='''which transformer backbone to use, current available options are:
                        deit_tiny_patch16_224;
                        deit_small_patch16_224;
                        deit_base_patch16_224;
                        ''')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true')
    parser.add_argument('--lwf', dest='lwf', action='store_true', help='add 2D pretraining task guidance')
    parser.add_argument('--reweighted', dest='reweighted', action='store_true')
    parser.add_argument('--head', default='default', type=str, help='either or not use different classifier (other than nn.Linear)')
    parser.add_argument('--embed-layer', type=str, default='VoxelEmbed', help='which way to embed the data')
    parser.add_argument('--cell-size', type=int, default=16, help='stride of CNN layer')
    parser.add_argument('--patch-size', type=int, default=8, help='tokenized sqeuence size (1 axis)')
    
    parser.add_argument('--pos-embedding', type=str, default = 'default', help='different positional embedding')
    parser.add_argument('--dist-url', type=str, default='localhost', help='ip for address')
    parser.add_argument('--lr', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr-step-size', type=float, default=20, help='learning rate')
    parser.add_argument('--lr-gamma', type=float, default=0.5, help='learning rate')

    

    parser.set_defaults(pretrained=False)
    parser.set_defaults(reweighted=False)
    parser.set_defaults(lwf=False)
    args = parser.parse_args()
    args.manualSeed = 9
    if "WORLD_SIZE" in os.environ:
        args.world_size = int(os.environ["WORLD_SIZE"])
        print("We have total of {} nodes involving.".format(args.world_size))
    else:
        args.world_size = args.gpus * args.nodes
    if args.dist_url == 'localhost':
        os.environ['MASTER_ADDR'] = args.dist_url
    else:
        os.environ['MASTER_ADDR'] = str(os.system("hostname -I | awk \'{print $1}\'"))
    os.environ['MASTER_PORT'] = args.port

    args.dist_url = "env://"
    args.slurm = False
    mp.spawn(train, nprocs= args.gpus, args=(args,), join=True)

