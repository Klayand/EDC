import os
import sys
import math
import time
import shutil
import argparse
import numpy as np
import wandb
import timm
import torch.distributed as dist

import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from torch.optim.lr_scheduler import LambdaLR
import torch.multiprocessing as mp
from prefetch_generator import BackgroundGenerator
from torch.utils.data import DataLoader
from utils import AverageMeter, accuracy, get_parameters

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

sys.path.append('../')
from relabel.utils_fkd import ImageFolder_FKD_MIX, ComposeWithCoords, RandomResizedCropWithCoords, \
    RandomHorizontalFlipWithRes, mix_aug

# It is imported for you to access and modify the PyTorch source code (via Ctrl+Click), more details in README.md
from torch.utils.data._utils.fetch import _MapDatasetFetcher

import torch
import torch.nn as nn


def cosine_similarity(a, b, eps=1e-5):
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + eps)


def pearson_correlation(a, b, eps=1e-5):
    return cosine_similarity(a - a.mean(1).unsqueeze(1),
                             b - b.mean(1).unsqueeze(1), eps)


def inter_class_relation(y_s, y_t):
    return 1 - pearson_correlation(y_s, y_t).mean()


def intra_class_relation(y_s, y_t):
    return inter_class_relation(y_s.transpose(0, 1), y_t.transpose(0, 1))


class DISTLoss(nn.Module):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, beta=2, gamma=2, tem=4):
        super(DISTLoss, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.tem = tem

    def forward(self, logits_student, logits_teacher):
        y_s = (logits_student / self.tem).softmax(dim=1)
        y_t = (logits_teacher / self.tem).softmax(dim=1)
        inter_loss = (self.tem ** 2) * inter_class_relation(y_s, y_t)
        intra_loss = (self.tem ** 2) * intra_class_relation(y_s, y_t)
        loss_kd = self.beta * inter_loss + self.gamma * intra_loss

        return loss_kd


def get_args():
    parser = argparse.ArgumentParser("FKD Training on ImageNet-1K")
    parser.add_argument('--batch-size', type=int,
                        default=1024, help='batch size')
    parser.add_argument('--gradient-accumulation-steps', type=int,
                        default=1, help='gradient accumulation steps for small gpu memory')
    parser.add_argument('--start-epoch', type=int,
                        default=0, help='start epoch')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--epochs', type=int, default=300, help='total epoch')
    parser.add_argument('-j', '--workers', default=0, type=int,
                        help='number of data loading workers')
    parser.add_argument('--loss-type', type=str, default="kl",
                        help='the type of the loss function')
    parser.add_argument('--train-dir', type=str, default=None,
                        help='path to training dataset')
    parser.add_argument('--val-dir', type=str,
                        default='/path/to/imagenet/val', help='path to validation dataset')
    parser.add_argument('--output-dir', type=str,
                        default='./save/1024', help='path to output dir')
    parser.add_argument('--cos', default=False,
                        action='store_true', help='cosine lr scheduler')
    parser.add_argument('--sgd', default=False,
                        action='store_true', help='sgd optimizer')
    parser.add_argument('-lr', '--learning-rate', type=float,
                        default=1.024, help='sgd init learning rate')  # checked
    parser.add_argument('--momentum', type=float,
                        default=0.875, help='sgd momentum')  # checked
    parser.add_argument('--weight-decay', type=float,
                        default=3e-5, help='sgd weight decay')  # checked
    parser.add_argument('--adamw-lr', type=float,
                        default=0.001, help='adamw learning rate')
    parser.add_argument('--adamw-weight-decay', type=float,
                        default=0.01, help='adamw weight decay')
    parser.add_argument('--ce-weight', type=float,
                        default=0.1, help='the weight og cross-entropy loss')
    parser.add_argument('--gpu-id', type=str,
                        default="0,1", help='the id of gpu used')
    parser.add_argument('--model', type=str,
                        default='resnet18', help='student model name')

    parser.add_argument('--keep-topk', type=int, default=1000,
                        help='keep topk logits for kd loss')
    parser.add_argument('-T', '--temperature', type=float,
                        default=3.0, help='temperature for distillation loss')
    parser.add_argument('--fkd-path', type=str,
                        default=None, help='path to fkd label')
    parser.add_argument('--wandb-project', type=str,
                        default='Temperature', help='wandb project name')
    parser.add_argument('--wandb-api-key', type=str,
                        default=None, help='wandb api key')
    parser.add_argument('--mix-type', default=None, type=str,
                        choices=['mixup', 'cutmix', None], help='mixup or cutmix or None')
    parser.add_argument('--fkd_seed', default=42, type=int,
                        help='seed for batch loading sampler')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')

    args = parser.parse_args()

    args.mode = 'fkd_load'
    return args


def main():
    args = get_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    port_id = 10002 + np.random.randint(0, 1000)
    args.dist_url = 'tcp://127.0.0.1:' + str(port_id)
    args.distributed = True
    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node * args.world_size
    torch.multiprocessing.set_start_method('spawn')
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))


def main_worker(gpu, ngpus_per_node, args):
    wandb.login(key=args.wandb_api_key)
    wandb.init(project=args.wandb_project, name=args.output_dir.split('/')[-1])
    args.rank = gpu
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)
    if not torch.cuda.is_available():
        raise Exception("need gpu to train!")

    assert os.path.exists(args.train_dir)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Data loading
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_dataset = ImageFolder_FKD_MIX(
        fkd_path=args.fkd_path,
        mode=args.mode,
        seed=args.fkd_seed,
        args_epoch=args.epochs,
        args_bs=args.batch_size,
        root=args.train_dir,
        transform=ComposeWithCoords(transforms=[
            RandomResizedCropWithCoords(size=224,
                                        scale=(0.08, 1),
                                        interpolation=InterpolationMode.BILINEAR),
            RandomHorizontalFlipWithRes(),
            transforms.ToTensor(),
            normalize,
        ]))

    generator = torch.Generator()
    generator.manual_seed(args.fkd_seed)
    grad_scaler = torch.cuda.amp.GradScaler()
    train_loader = DataLoaderX(
        train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    # load validation data
    val_loader = DataLoaderX(
        datasets.ImageFolder(args.val_dir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=int(args.batch_size / 4), shuffle=False,
        num_workers=args.workers, pin_memory=True)
    print('load data successfully')

    # load student model
    print("=> loading student model '{}'".format(args.model))
    if args.model in torchvision.models.__dict__:
        model = torchvision.models.__dict__[args.model](pretrained=False)
    else:
        model = timm.create_model(args.model, num_classes=1000, pretrained=False)
    torch.cuda.set_device(gpu)
    model = DDP(model.cuda(gpu), device_ids=[gpu], output_device=gpu)
    model.train()

    if args.sgd:
        optimizer = torch.optim.SGD(get_parameters(model),
                                    lr=args.learning_rate,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(get_parameters(model),
                                      lr=args.adamw_lr,
                                      weight_decay=args.adamw_weight_decay)

    if args.cos == True:
        scheduler = LambdaLR(optimizer,
                             lambda step: 0.5 * (
                                     1. + math.cos(math.pi * step / (3*args.epochs/2))) if step <= (3*args.epochs/2) else 0,
                             last_epoch=-1)
    else:
        scheduler = LambdaLR(optimizer,
                             lambda step: (1.0 - step / (3*args.epochs/2)) if step <= (3*args.epochs/2) else 0, last_epoch=-1)

    args.best_acc1 = 0 # 31.4% -> 34.4% (background)
    args.optimizer = optimizer
    args.scheduler = scheduler
    args.train_loader = train_loader
    args.val_loader = val_loader

    for epoch in range(args.start_epoch, args.epochs):
        print(f"\nEpoch: {epoch}")

        global wandb_metrics
        wandb_metrics = {}

        train(model, args, epoch, gpu, ngpus_per_node, scaler=grad_scaler)

        if epoch % 10 == 0 or epoch == args.epochs - 1:
            top1 = validate(model, args, epoch)
        else:
            top1 = 0

        wandb.log(wandb_metrics)

        scheduler.step()

        # remember best acc@1 and save checkpoint
        is_best = top1 > args.best_acc1
        args.best_acc1 = max(top1, args.best_acc1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc1': args.best_acc1,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }, is_best, output_dir=args.output_dir)

    wandb.finish()


def adjust_bn_momentum(model, iters):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = 1 / iters


def train(model, args, epoch=None, gpu=0, ngpus_per_node=1, scaler=None):
    objs = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    optimizer = args.optimizer
    scheduler = args.scheduler
    loss_function_kl = nn.KLDivLoss(reduction='batchmean')
    loss_function_dist = DISTLoss(tem=args.temperature)

    model.train()
    t1 = time.time()
    args.train_loader.dataset.set_epoch(epoch)
    for batch_idx, batch_data in enumerate(args.train_loader):
        images, target, flip_status, coords_status = batch_data[:4]
        mix_index, mix_lam, mix_bbox, soft_label = batch_data[4:]

        images = images.cuda()
        target = target.cuda()
        soft_label = soft_label.cuda().float()  # convert to float32
        right = torch.all((mix_bbox[0].float() - mix_bbox[0].float().mean() + 1).bool()) \
                & torch.all((mix_bbox[1].float() - mix_bbox[1].float().mean() + 1).bool()) \
                & torch.all((mix_bbox[2].float() - mix_bbox[2].float().mean() + 1).bool()) \
                & torch.all((mix_bbox[3].float() - mix_bbox[3].float().mean() + 1).bool())
        if right.item() == 0:
            raise KeyError("In a batch, the mix_bbox should be same!")
        mix_bbox = [mix_bbox[0][0], mix_bbox[1][0], mix_bbox[2][0], mix_bbox[3][0]]
        images, _, _, _ = mix_aug(images, args, mix_index, mix_lam, mix_bbox)

        optimizer.zero_grad()
        assert args.batch_size % (args.gradient_accumulation_steps * ngpus_per_node) == 0
        small_bs = args.batch_size // (args.gradient_accumulation_steps * ngpus_per_node)

        # images.shape[0] is not equal to args.batch_size in the last batch, usually
        if batch_idx == len(args.train_loader) - 1:
            accum_step = math.ceil(images.shape[0] / small_bs / ngpus_per_node)
        else:
            accum_step = args.gradient_accumulation_steps

        for accum_id in range(0, accum_step * ngpus_per_node, ngpus_per_node):
            if accum_id == accum_step * ngpus_per_node - ngpus_per_node and batch_idx == len(args.train_loader) - 1:
                last_number = (images.shape[0] - small_bs * (accum_step * ngpus_per_node - ngpus_per_node)) // 2
                partial_images = images[
                                 accum_id * small_bs + last_number * gpu: accum_id * small_bs + last_number * (gpu + 1)]
                partial_target = target[
                                 accum_id * small_bs + last_number * gpu: accum_id * small_bs + last_number * (gpu + 1)]
                partial_soft_label = soft_label[
                                     accum_id * small_bs + last_number * gpu: accum_id * small_bs + last_number * (
                                                 gpu + 1)]
            else:
                partial_images = images[(accum_id + gpu) * small_bs: (accum_id + gpu + 1) * small_bs]
                partial_target = target[(accum_id + gpu) * small_bs: (accum_id + gpu + 1) * small_bs]
                partial_soft_label = soft_label[(accum_id + gpu) * small_bs: (accum_id + gpu + 1) * small_bs]
            with torch.cuda.amp.autocast(enabled=False):
                output = model(partial_images).float()
            prec1, prec5 = accuracy(output, partial_target, topk=(1, 5))

            if args.loss_type == "kl":
                output = F.log_softmax(output / args.temperature, dim=1)
                partial_soft_label = F.softmax(partial_soft_label / args.temperature, dim=1)
                loss = loss_function_kl(output, partial_soft_label)
            elif args.loss_type == "dist":
                loss = loss_function_dist(output, partial_soft_label)
            elif args.loss_type == "mse_gt":
                loss = F.mse_loss(output, partial_soft_label) + F.cross_entropy(output, partial_target) * args.ce_weight
            else:
                raise NotImplementedError
            # loss = loss * args.temperature * args.temperature
            loss = loss / args.gradient_accumulation_steps
            loss.backward()
            # scaler.scale(loss).backward()
            n = partial_images.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)
        optimizer.step()
        # scaler.step(optimizer)
        # scaler.update()

    metrics = {
        "train/loss": objs.avg,
        "train/Top1": top1.avg,
        "train/Top5": top5.avg,
        "train/lr": scheduler.get_last_lr()[0],
        "train/epoch": epoch, }
    wandb_metrics.update(metrics)

    printInfo = 'TRAIN Iter {}: lr = {:.6f},\tloss = {:.6f},\t'.format(epoch, scheduler.get_last_lr()[0], objs.avg) + \
                'Top-1 err = {:.6f},\t'.format(100 - top1.avg) + \
                'Top-5 err = {:.6f},\t'.format(100 - top5.avg) + \
                'train_time = {:.6f}'.format((time.time() - t1))
    print(printInfo)
    t1 = time.time()


def validate(model, args, epoch=None):
    objs = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    loss_function = nn.CrossEntropyLoss()

    model.eval()
    t1 = time.time()
    with torch.no_grad():
        for data, target in args.val_loader:
            target = target.type(torch.LongTensor)
            data, target = data.cuda(), target.cuda()

            output = model(data)
            loss = loss_function(output, target)

            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            n = data.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

    logInfo = 'TEST Iter {}: loss = {:.6f},\t'.format(epoch, objs.avg) + \
              'Top-1 err = {:.6f},\t'.format(100 - top1.avg) + \
              'Top-5 err = {:.6f},\t'.format(100 - top5.avg) + \
              'val_time = {:.6f}'.format(time.time() - t1)
    print(logInfo)

    metrics = {
        'val/loss': objs.avg,
        'val/top1': top1.avg,
        'val/top5': top5.avg,
        'val/epoch': epoch,
    }
    wandb_metrics.update(metrics)

    return top1.avg


def save_checkpoint(state, is_best, output_dir=None, epoch=None):
    if epoch is None:
        path = output_dir + '/' + 'checkpoint.pth.tar'
    else:
        path = output_dir + f'/checkpoint_{epoch}.pth.tar'
    torch.save(state, path)

    if is_best:
        path_best = output_dir + '/' + 'model_best.pth.tar'
        shutil.copyfile(path, path_best)


if __name__ == "__main__":
    main()
