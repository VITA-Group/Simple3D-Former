# Simple3D-Former
This is the official repo for [Can We Solve 3D Vision Tasks Starting from A 2D Vision Transformer?](https://arxiv.org/pdf/2209.07026.pdf)

## Perquisitive

### Environment Setup
It is tested in python 3.7 with the following packages as minimal support:

*   einops==0.3.0
*   linformer==0.2.1
*   torch==1.7.1
*   torchvision==0.8.2
*   tqdm
*   hydra==2.5
*   hydra-core==1.1.1
*   omegaconf==2.1.1
*   h5py
*   plyfile

In addition, since [DeIT](https://github.com/facebookresearch/deit) heavily depends on timm, make sure you have

``` pip install timm==0.3.2  ```

We provide a simple ```requirements.txt``` to install the library (with full package lists provided) with pip as well, by excecuting;

```pip install -r requirements.txt ```


### DataSet Preparation

Currently ShapeNetV2/ModelNet40/ShapeNetPart are required. The teacher dataset is the ImageNet validation set (in ImageNet 1K).

*   ShapeNetV2: Download it from here: https://shapenet.cs.stanford.edu/shapenet/obj-zip/ShapeNetCore.v2.zip;
*   ModelNet40: Download it from here: https://modelnet.cs.princeton.edu/ModelNet40.zip;
*   ModelNet40 point cloud samples: Download it from here: https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip;
*   ShapeNetPart: Download it from here: https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip;
*   ImageNet 1K: Download it from kaggle: https://www.kaggle.com/c/imagenet-object-localization-challenge/data;
*   ScanObjectNN: Download it from here: http://103.24.77.34/scanobjectnn/h5_files.zip

Then extract all files in ```./data/``` folder in current project. You can modify config files under ```./config/``` for a specific data location (especially if you are downloading full ImageNet instead of this particular subset). In addition, after downloading ModelNet40, you need to create all *.binvox file by doing:

``` python 
cd data/
python binvox_convert.py ModelNet40/ --remove-all-dupes
```

## How to run

### Voxel Classification

Run ```train_cls_voxel.py``` script. The default usage of this script is:

```python
python train_cls_voxel.py 
```

To reproduce, one needs to enumerate configurations of backbone and positional embeddings, as well as dataset. We provide two examples of ModelNet40 and ShapeNetV2 respectively:

```python
python train_cls_voxel.py --data-root ./data/ModelNet40 --batchSize 64 --pretrained --lwf --epochs 100 --gpus 1 --dataset ModelNet40 --transformer-name deit_small_patch16_224 --outf ./cls/ --pos-embedding default --embed-layer VoxelEmbed --cell-size 6 --patch-size 5 --lr 1e-3
```

```python
python train_cls_voxel.py --data-root ./data/ShapeNetCore_v2 --batchSize 64 --pretrained --lwf --epochs 100 --gpus 1 --dataset ShapeNetV2 --transformer-name deit_base_patch16_224 --outf ./cls/ --pos-embedding group_embed --embed-layer VoxelEmbed_no_average --cell-size 9 --patch-size 14 --lr 1e-3
```

The configuration ```--pos-embedding``` and ```--embed-layer``` shall match. Three different tokenized scheme refers to:

* Naive Tokenize: ```--pos-embedding default --embed-layer VoxelEmbed```;
* 2D Projection: ```--pos-embedding default --embed-layer VoxelEmbed_no_average```;
* Group Embedding: ```--pos-embedding group_embed --embed-layer VoxelEmbed_no_average```;

### Point Cloud Tasks

These part of scripts is adapted from https://github.com/qq456cvb/Point-Transformers One can modify ```./config``` files to adjust parameters. To run the script, simply run:

*  Point Cloud Classification: ```python train_cls.py``` or ```python train_cls_scanobjectnn.py```;
*  Point Cloud Part Segmentation: ```python train_partseg.py```;
*  Point Cloud Part Segmentation with 2D knowledge: ```python train_partseg_lwf.py```
*  Point Cloud Object Detection (TBD)
