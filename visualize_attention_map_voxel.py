# https://github.com/samiraabnar/attention_flow
# https://github.com/google-research/vision_transformer/issues/27
# https://github.com/google-research/vision_transformer/issues/18
# https://github.com/faustomorales/vit-keras/blob/65724adcfd3979067ce24734f08df0afa745637d/vit_keras/visualize.py#L7-L45
# https://github.com/jeonsworld/ViT-pytorch/blob/main/visualize_attention_map.ipynb


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
from data.modelnet10 import ModelNet10
from data.modelnet40 import ModelNet40
from data.shapenet_v2 import ShapeNetV2
from linformer import Linformer
from global_var import *

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import warnings
from datetime import date
from datetime import timedelta

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
#import cv2
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

activation = {}

torch.hub.set_dir('./cls')


VALID_EMBED_LAYER={
    'VoxelEmbed': VoxelEmbed(),
    'VoxelEmbed_no_average': VoxelEmbed_no_average(),
    'VoxelEmbed_Hybrid': VoxelEmbed_Hybrid(),
    'VoxelEmbed_Hybrid_no_average': VoxelEmbed_Hybrid_no_average(),
    'VoxelEmbed_14': VoxelEmbed(cell_size=9, patch_size=14),
    'VoxelEmbed_no_average_14': VoxelEmbed_no_average(cell_size=9, patch_size=14),
}


def evaluate_attention_map(args):

    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    CLASSES= CLASSES_SHAPENET
    N_CLASSES=len(CLASSES)

    
    dataset = ShapeNetV2(data_root=args.data_root, n_classes=N_CLASSES, idx2cls=CLASSES)
    try:
        embedding = VALID_EMBED_LAYER[args.embed_layer]   
    except:
        print("Unknown type of 3D data embedding!")
        raise ValueError


    model = Feature3D_ViT2D_V2(embed_layer = embedding,
        n_classes=N_CLASSES, transformer_backbone=args.transformer_name, 
        pretrained=False, pos_embedding=args.pos_embedding, head=args.head).to(device)
    
    #remove DDP keys
    

    train_size = int(0.8*len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(args.manualSeed))

    train_dataloader = DataLoader(train_dataset, batch_size=args.batchSize, shuffle=False, num_workers=4, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batchSize, shuffle=False, num_workers=4, pin_memory=True)

    if args.pretrained:
        output_folder =os.path.join(args.outf, 'visualization/%s_%s_%s_%s/' % (args.model_name, args.embed_layer, args.pos_embedding, args.transformer_name))
    else:
        output_folder =os.path.join(args.outf, 'visualization/%s_%s_%s_%s_no_pretrain/' % (args.model_name, args.embed_layer, args.pos_embedding, args.transformer_name))
    os.makedirs(output_folder, exist_ok=True)

    

    if args.model != '':
        state_dict = torch.load(args.model, map_location=device)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        print(state_dict.keys())
        for k, v in state_dict.items():
            name = k.replace("module.", "")# remove module.
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    # sanity check

    model.eval()
    print("Start attention map computation")
    
    #case = train_dataset[0]
    #show_attention_map(model, torch.Tensor(case['voxel']).float().to(device), device, output_folder, id=case['model_id'])
    case = train_dataset[10000]
    print(case['model_id'])
    show_attention_map(model, torch.Tensor(case['voxel']).float().to(device), device, output_folder, id=case['model_id'])
    #case = test_dataset[0]
    #show_attention_map(model, torch.Tensor(case['voxel']).float().to(device), device, output_folder, id=case['model_id'])
    #case = train_dataset[-1]
    #show_attention_map(model, torch.Tensor(case['voxel']).float().to(device), device, output_folder, id=case['model_id'])

def get_attn_softmax(name):
    def hook(model, input, output):
        with torch.no_grad():
            input = input[0]
            B, N, C = input.shape
            qkv = (
                model.qkv(input)
                .detach()
                .reshape(B, N, 3, model.num_heads, C // model.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
            q, k, v = (
                qkv[0],
                qkv[1],
                qkv[2],
            )  # make torchscript happy (cannot use tensor as tuple)
            attn = (q @ k.transpose(-2, -1)) * model.scale
            attn = attn.softmax(dim=-1)
            activation[name] = attn

    return hook


# expects timm vis transformer model
def add_attn_vis_hook(model):
    for idx, module in enumerate(list(model.blocks.children())):
        module.attn.register_forward_hook(get_attn_softmax(f"attn{idx}"))

def cuboid_data(o, size=(1,1,1)):
    X = [[[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0]],
         [[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]],
         [[1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]],
         [[0, 0, 1], [0, 0, 0], [0, 1, 0], [0, 1, 1]],
         [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]],
         [[0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1]]]
    X = np.array(X).astype(float)
    for i in range(3):
        X[:,:,i] *= size[i]
    X += np.array(o)
    return X

def get3Dposition(voxels, sizes=None, colors=None, **kwargs):
    positions = []
    voxels = voxels.squeeze()
    for i in range(voxels.shape[0]):
        for j in range(voxels.shape[1]):
            for k in range(voxels.shape[2]):
                if voxels[i][j][k] > 0:
                    positions.append(np.array([i,j,k]))
    positions = np.c_[positions] / (1.0 * voxels.shape[0])
    print(positions.shape)
    print("Position calculation complete!")

    return positions

def get_mask(im, att_mat, device):
    # Average the attention weights across all heads.
    # att_mat,_ = torch.max(att_mat, dim=1)
    att_mat = torch.mean(att_mat, dim=1)

    # To account for residual connections, we add an identity matrix to the
    # attention matrix and re-normalize the weights.
    residual_att = torch.eye(att_mat.size(1)).to(device)
    aug_att_mat = att_mat + residual_att
    #aug_att_mat = att_mat
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

    # Recursively multiply the weight matrices
    joint_attentions = torch.zeros(aug_att_mat.size()).to(device)
    joint_attentions[0] = aug_att_mat[0]

    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])
        
    # Attention from the output token to the input space.
    v = joint_attentions[-1]
    grid_size = int(np.sqrt(aug_att_mat.size(-1)))
    mask = v[0, 1:].reshape(grid_size, grid_size).detach().cpu().numpy()
    # mask = cv2.resize(mask / mask.max(), im.size)[..., np.newaxis]
    # result = (mask * im).astype("uint8")

    result = mask
    return result, joint_attentions, grid_size

def plot_layer_attention_map(model, result, voxel, device, output_folder, index=0):
    pos_embedding =  model.pos_embed_type
    if pos_embedding is None or pos_embedding=="default" or pos_embedding=="group_embed" or pos_embedding=="weight_sharing":
        plt.figure()
        
        plt.imshow(result)
        plt.colorbar()
        if index==0:
            plt.title('Final Attention Map')
            plt.savefig("%s/attn_final.png" % output_folder)
        else:
            plt.title('Layer %dth Attention Map' % (index))
            plt.savefig("%s/attn_%d.png" % (output_folder, index))
       
    elif pos_embedding=="no_embed":
        pass

    if pos_embedding=="group_embed":
        pass
    if pos_embedding=="weight_sharing":
        pass


def show_attention_map(model, voxel, device, output_folder, id='default'):
    add_attn_vis_hook(model)
    
    logits = model(voxel.unsqueeze(0))
    voxel = torch.nn.functional.interpolate(voxel.unsqueeze(0), size=(32,32,32), mode="trilinear").squeeze()

    attn_weights_list = list(activation.values())

    result, joint_attentions, grid_size = get_mask(voxel ,torch.cat(attn_weights_list), device)
    
    output_folder = os.path.join(output_folder, id) 
    os.makedirs(output_folder, exist_ok=True)
    fig = plt.figure()
    ax1 = fig.add_subplot(projection='3d')
    #ax1.set_aspect('equal')
    pc = get3Dposition(voxel, edgecolor="k")
    ax1.set_title('Original')
    ax1.set_xlabel('y')
    ax1.set_ylabel('z')
    ax1.set_zlabel('x')
    for each in pc:
        ax1.scatter(each[1], each[2], each[0])

    plt.savefig("%s/original_voxel.png" % (output_folder))
    #_ = ax1.imshow(im)

    plot_layer_attention_map(model,result, voxel, device, output_folder)
    
    probs = torch.nn.Softmax(dim=-1)(logits)
    print(probs)
    top5 = torch.argsort(probs, dim=-1, descending=True)
    print("Prediction Label and Attention Map!\n")
    for idx in top5[0, :5]:
        print(f'{probs[0, idx.item()]:.5f} : {idx.item()} ', end='')

    for i, v in enumerate(joint_attentions):
        # Attention from the output token to the input space.
        mask = v[0, 1:].reshape(grid_size, grid_size).detach().cpu().numpy()
        #mask = cv2.resize(mask / mask.max(), im.size)[..., np.newaxis]
        #result = (mask * im).astype("uint8")
        result = mask

        plot_layer_attention_map(model,result, voxel, device, output_folder, index=i+1)
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, default='/mnt/data/yiwang/ShapeNetCore_v2', help="dataset path")
    parser.add_argument('--batchSize', type=int, default=4, help='input batch size')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
    parser.add_argument('--outf', type=str, default='./cls', help='output folder')
    parser.add_argument('--model', type=str, default='cls/11_09_2021/Voxel3D_2DPretrain/VoxelEmbed_default/deit_base_patch16_224/epoch_best.pth', help='checkpoint model path')
    parser.add_argument('--dataset', type=str, default='ShapeNetV2', help='which dataset to be used')
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N', help='number of nodes')
    parser.add_argument('-g', '--gpus', default=2, type=int, help='number of gpus per node')
    parser.add_argument('-rank', '--rank', default=0, type=int, help='ranking within the nodes')
    parser.add_argument('--port', default='12455', type=str, metavar='P', help='port number for parallel')
    parser.add_argument('--epochs', default=2, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--model-name', type=str, default='Voxel3D_2DPretrain', help='which model to use')
    parser.add_argument('--transformer-name', type=str, default='deit_base_patch16_224', help='''which transformer backbone to use, current available options are:
                        deit_tiny_patch16_224; 
                        deit_small_patch16_224; 
                        deit_base_patch16_224; 
                        deit_tiny_distilled_patch16_224;
                        deit_small_distilled_patch16_224; 
                        deit_base_distilled_patch16_224;
                        ''')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true')
    parser.add_argument('--reweighted', dest='reweighted', action='store_true')
    parser.add_argument('--head', default='AMSoftmax', type=str, help='either or not use different classifier (other than nn.Linear)')
    parser.add_argument('--embed-layer', type=str, default='VoxelEmbed', help='which way to embed the data')
    parser.add_argument('--pos-embedding', type=str, default = 'default', help='different positional embedding')
    parser.add_argument('--dist-url', type=str, default='localhost', help='ip for address')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    
    parser.set_defaults(pretrained=True)
    parser.set_defaults(reweighted=False)
    args = parser.parse_args()
    args.manualSeed = 9

    evaluate_attention_map(args)