
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
from data.modelnet10 import ModelNet10
from data.modelnet40 import ModelNet40
from data.shapenet_v2 import ShapeNetV2
from models.embed_layer_3d_modality import *
from global_var import *
from models.vip_3d import vip3d_s7,vip3d_s14,vip3d_m7,vip3d_l7
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import warnings
from datetime import date
from datetime import timedelta
import pytorch_warmup as warmup

from timm.models import load_checkpoint, create_model, resume_checkpoint, convert_splitbn_model
# from timm.utils import *

today = date.today()

# dd_mm_YY
day = today.strftime("%d_%m_%Y")

VALID_EMBED_LAYER={
    'VoxelEmbed': VoxelEmbed(),
    'VoxelEmbed_m40_vip_s7': VoxelEmbed_no_average(embed_dim=192, voxel_size=32, cell_size=4),
    'VoxelEmbed_m40_vip_s14': VoxelEmbed_no_average(embed_dim=384, voxel_size=32, cell_size=4),
    'VoxelEmbed_m40_vip_m7': VoxelEmbed_no_average(embed_dim=256, voxel_size=32, cell_size=4),
    'VoxelEmbed_m40_vip_l7': VoxelEmbed_no_average(embed_dim=256, voxel_size=32, cell_size=4),
    'VoxelEmbed_vip_s7': VoxelEmbed_no_average(embed_dim=192),
    'VoxelEmbed_vip_s14': VoxelEmbed_no_average(embed_dim=384),
    'VoxelEmbed_vip_m7': VoxelEmbed_no_average(embed_dim=256),
    'VoxelEmbed_vip_l7': VoxelEmbed_no_average(embed_dim=256),
}



torch.hub.set_dir('./cls')


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
    print(f"Running basic mp on {args.dist_url}, node rank {args.rank}, gpu id {gpu}.")
    setup(rank, args.world_size, args.dist_url)
    print(f'Finish setup the process')

    # TODO: Fix ShapeNetV1 support
    if args.dataset == "ShapeNetV2":
        shapenet_version = 2
        dataroot = SHAPENETV2_ROOT
        class_number = CLASSES_SHAPENET_NUMBER
        CLASSES= CLASSES_SHAPENET
        N_CLASSES=len(CLASSES)
        dataset = ShapeNetV2(data_root=args.data_root, n_classes=N_CLASSES, idx2cls=CLASSES)
    elif args.dataset == "ShapeNetV1":
        shapenet_version = 1
        dataroot = SHAPENETV1_ROOT
        class_number = CLASSES_SHAPENET_NUMBER_V1
        CLASSES= CLASSES_SHAPENET
        N_CLASSES=len(CLASSES)
        dataset = ShapeNetV2(data_root=args.data_root, n_classes=N_CLASSES, idx2cls=CLASSES)
    elif args.dataset == "ModelNet40":
        dataroot = ModelNet40_ROOT
        CLASSES= CLASSES_ModelNet40
        N_CLASSES=len(CLASSES)
        train_dataset = ModelNet40(data_root=ModelNet40_ROOT, n_classes=N_CLASSES, idx2cls=CLASSES, split='train')
        test_dataset = ModelNet40(data_root=ModelNet40_ROOT, n_classes=N_CLASSES, idx2cls=CLASSES, split='test')




    try:
        embedding = VALID_EMBED_LAYER[args.embed_layer]
    except:
        print("Unknown type of 3D data embedding!")
        raise ValueError
    model = create_model(
        args.model_name,
        pretrained=args.pretrained,
        num_classes=N_CLASSES,
        in_chans=1,
        #drop_rate=args.drop,
        drop_path_rate=0.1,
        #drop_block_rate=args.drop_block,
        #global_pool=args.gp,
        #bn_tf=args.bn_tf,
        #bn_momentum=args.bn_momentum,
        #bn_eps=args.bn_eps,
        checkpoint_path_2d=args.checkpoint_path_2d,
        img_size=128 if "ShapeNet" in args.dataset else 32,
        embed_layer = embedding,
        pos_embedding=args.pos_embedding,
        device = device
        ).to(device)
    # torch.autograd.set_detect_anomaly(True)
    # model = vip3d_s14(
    #         embed_layer = embedding,
    #         num_classes=N_CLASSES,
    #         in_chans=1,
    #         pos_embedding=args.pos_embedding).to(device)
    # print(model)
    model = DDP(model,
    device_ids = [gpu],
    output_device= gpu,
    broadcast_buffers=False,
    find_unused_parameters=True
    )

    if args.dataset == "ShapeNetV2":
        train_size = int(0.8*len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(args.manualSeed))

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=args.world_size,
        rank=rank)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batchSize, shuffle=False, num_workers=1, pin_memory=True, sampler=train_sampler)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batchSize, shuffle=True, num_workers=1, pin_memory=True)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.95)
    warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)

    if args.pretrained:
        output_folder =os.path.join(args.outf, '%s/%s/%s_%s' % (day, args.model_name, args.embed_layer, args.pos_embedding))
    else:
        output_folder =os.path.join(args.outf, '%s/%s_no_pretrain/%s_%s' % (day, args.model_name, args.embed_layer, args.pos_embedding))
    os.makedirs(output_folder, exist_ok=True)


    if args.model != '':
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

    print("Start training loop")
    for epoch in range(args.epochs):
        # Train
        for i, sample in tqdm(enumerate(train_dataloader, 0)):
            model.train()
            voxel, cls_idx = sample['voxel'], sample['cls_idx']

            voxel, cls_idx = voxel.to(device), cls_idx.to(device)
            voxel = voxel.float()
            optimizer.zero_grad()
            pred = model(voxel)
            loss = F.cross_entropy(pred, cls_idx)

            loss.backward()
            optimizer.step()

            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(cls_idx.data).cpu().sum()
            #print('[%d: %d/%d] train loss: %f accuracy: %f' %
            #       (epoch, i, int(len(dataset)/args.batchSize), loss.item(), correct.item() / float(args.batchSize)))
            #scheduler.step()
            scheduler.step(scheduler.last_epoch+1)
            warmup_scheduler.dampen()


        if rank == 0:

            total_correct = 0
            total_testset = 0
            class_correct = torch.zeros(N_CLASSES)
            class_testset = torch.zeros(N_CLASSES)
            # Test
            with torch.no_grad():
                for i, sample in tqdm(enumerate(test_dataloader, 0)):
                    model.eval()
                    voxel, cls_idx = sample['voxel'], sample['cls_idx']
                    voxel, cls_idx = voxel.to(device), cls_idx.to(device)

                    voxel = voxel.float()
                    pred = model(voxel)
                    pred_choice = pred.data.max(1)[1]
                    correct = pred_choice.eq(cls_idx.data).cpu().sum()
                    total_correct += correct.item()
                    total_testset += voxel.shape[0]
                    corrects = pred_choice.eq(cls_idx.data).cpu()
                    for j in range(voxel.size()[0]):
                        class_correct[cls_idx[j]] += corrects[j].item()
                        class_testset[cls_idx[j]] += 1.0
                print("Total test samples: {}".format(total_testset))
                print("Epoch %d test accuracy %f" % (epoch, total_correct / float(total_testset)))
                if total_correct / float(total_testset) >= best_acc:
                    best_acc = total_correct / float(total_testset)
                    best_epoch = epoch
                    torch.save(model.state_dict(), '%s/epoch_best.pth' % output_folder)
                for i in range(N_CLASSES):
                    print("Class %s: %.3f" % (CLASSES[i], class_correct[i]/class_testset[i]))
        
        if args.world_size > 1:
            dist.barrier()

    if rank == 0:
        print("Best test accuracy: epoch %d test accuracy %f" % (best_epoch, best_acc))

    cleanup()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, default='/mnt/storage/datasets/ShapeNetCore_v2', help="dataset path")
    parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
    parser.add_argument('--outf', type=str, default='./cls', help='output folder')
    parser.add_argument('--model', type=str, default='', help='checkpoint model path')
    parser.add_argument('--dataset', type=str, default='ModelNet40', help='which dataset to be used')
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N', help='number of nodes')
    parser.add_argument('-g', '--gpus', default=4, type=int, help='number of gpus per node')
    parser.add_argument('-rank', '--rank', default=0, type=int, help='ranking within the nodes')
    parser.add_argument('--port', default='12455', type=str, metavar='P', help='port number for parallel')
    parser.add_argument('--epochs', default=50, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--model-name', type=str, default='vip3d_s7', help='which model to use')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true', default=False)
    parser.add_argument('--checkpoint-path-2d', type=str, default='', help="2d pretrained weight")
    parser.add_argument('--embed-layer', type=str, default='VoxelEmbed_m40_vip_s7', help='which way to embed the data')
    parser.add_argument('--pos-embedding', type=str, default = 'default', help='different positional embedding')
    parser.add_argument('--dist-url', type=str, default='localhost', help='ip for address')
    parser.add_argument('--lr', type=float, default=5e-2, help='learning rate')



    args = parser.parse_args()
    if args.checkpoint_path_2d=="default":
        args.checkpoint_path_2d=''
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

    if "SLURM_PROCID" in os.environ:
        jobid = os.environ["SLURM_JOBID"]
        hostfile = "./cls/dist_url." + jobid  + ".txt"
         # For some reason, TACC's SLURM_PROCID will always be 0
        args.rank = int(os.environ["SLURM_PROCID"])
        args.world_size = int(os.environ["SLURM_NPROCS"])
        args.slurm = True

        if args.rank == 0:
            import socket
            ip = socket.gethostbyname(socket.gethostname())
            port = find_free_port()
            args.dist_url = "tcp://{}:{}".format(ip, port)
            with open(hostfile, "w") as f:
                f.write(args.dist_url)
        else:
            import os
            import time
            while not os.path.exists(hostfile):
                time.sleep(1)
            with open(hostfile, "r") as f:
                args.dist_url = f.read()

        train(args.rank % args.gpus, args)
    else:
        args.dist_url = "env://"
        args.slurm = False
        mp.spawn(train, nprocs= args.gpus, args=(args,), join=True)

