import torch
import torch.nn as nn
import random
from torch.utils.data import random_split
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

from pytorch3d.datasets import ShapeNetCore
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PointLights, 
    RasterizationSettings, 
)

import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

import sys
sys.path.insert(0,'..')
from global_var import SHAPENETV2_ROOT, SHAPENETV1_ROOT

def generate_fixed_view_rendering():
    dataset= ShapeNetCore(SHAPENETV2_ROOT, version=2, texture_resolution=6)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    R, T = look_at_view_transform(1.0, 1.0, 90)
    cameras = FoVPerspectiveCameras(R=R, T=T, device=device)
    raster_settings = RasterizationSettings(image_size=224, faces_per_pixel=10, cull_backfaces=True, cull_to_frustum=True)
    lights = PointLights(location=torch.tensor([0.0, 1.0, -2.0], device=device)[None],device=device)

    os.makedirs('.tmp/', exist_ok=True)

    for i in tqdm(range(len(dataset))):
        filename = '.tmp/%s' % dataset.model_ids[i]
        filename = filename+'.png'
        if os.path.exists(filename):
            continue
        image = dataset.render(
            model_ids=dataset.model_ids[i],
            device=device,
            cameras=cameras,
            raster_settings=raster_settings,
            lights=lights,
        )
        save_image(image[0, ..., :3].permute(2,0,1), filename)
        del image

def __setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def __cleanup():
    dist.destroy_process_group()

def __generate_fixed_view_rendering_process(rank,world_size,shapenet_version=2): 
    print(f"Running basic mp on rank {rank}.")

    device = torch.device("cuda:%d" % rank)
    torch.cuda.set_device(device)
    __setup(rank, world_size)

    if shapenet_version==2:
        dataset_root = SHAPENETV2_ROOT
    elif shapenet_version==1:
        dataset_root = SHAPENETV1_ROOT
    else:
        print("Wrong version number!")
    dataset= ShapeNetCore(dataset_root, version=shapenet_version, texture_resolution=6)

    R, T = look_at_view_transform(1.0, 1.0, 90)
    cameras = FoVPerspectiveCameras(R=R, T=T, device=device)
    raster_settings = RasterizationSettings(image_size=224, faces_per_pixel=10, cull_backfaces=True)
    lights = PointLights(location=torch.tensor([0.0, 1.0, -2.0], device=device)[None],device=device)

    
    
    bad_model = []

    for i in tqdm(range(len(dataset))):
        filepath = '.tmp/%s/' % dataset.synset_ids[i]
        os.makedirs(filepath, exist_ok=True)
        filename = filepath + dataset.model_ids[i] + '.png'

        if os.path.exists(filename):
            continue
        #print("Generating model rendering under %s" % filename)
        try:
            image = dataset.render(
                idxs=[i],
                device=device,
                cameras=cameras,
                raster_settings=raster_settings,
                lights=lights,
            )
        except:
            #print("Ignoring model %s" % filename)
            bad_model.append('%s/%s' % (dataset.synset_ids[i], dataset.model_ids[i]))
            continue
        # plt.imshow(image[0, ..., :3].cpu().numpy())
        save_image(image[0, ..., :3].permute(2,0,1), filename)
        del image

    for each in bad_model:
        print(each)
    __cleanup()

def generate_fixed_view_rendering_parallel(version):
    # read all possible devices
    world_size = torch.cuda.device_count()
    print("Use %d gpus." % world_size)
    mp.spawn(__generate_fixed_view_rendering_process,
             args=(world_size,version,),
             nprocs=world_size,
             join=True)




if __name__ == '__main__':
    generate_fixed_view_rendering('/mnt/storage/datasets/ShapeNetCore_v2')