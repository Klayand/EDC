'''This code is modified from https://github.com/liuzechun/Data-Free-NAS'''

import torch
from torch import distributed
import numpy as np
import torch.nn.functional as F
import os, sys, random
import einops
import torch.distributed as dist


def distributed_is_initialized():
    if distributed.is_available():
        if distributed.is_initialized():
            return True
    return False


def lr_policy(lr_fn):
    def _alr(optimizer, iteration, epoch):
        lr = lr_fn(iteration, epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    return _alr


def div_sixteen_mul(v):
    v = int(v)
    m = v % 16
    return int(v // 16 * 16) + int(m > 0) * 16


def lr_cosine_policy(base_lr, warmup_length, epochs):
    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            e = epoch - warmup_length
            es = epochs - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        return lr

    return lr_policy(_lr_fn)


def beta_policy(mom_fn):
    def _alr(optimizer, iteration, epoch, param, indx):
        mom = mom_fn(iteration, epoch)
        for param_group in optimizer.param_groups:
            param_group[param][indx] = mom

    return _alr


def mom_cosine_policy(base_beta, warmup_length, epochs):
    def _beta_fn(iteration, epoch):
        if epoch < warmup_length:
            beta = base_beta * (epoch + 1) / warmup_length
        else:
            beta = base_beta
        return beta

    return beta_policy(_beta_fn)


def clip(image_tensor, use_fp16=False):
    '''
    adjust the input based on mean and variance
    '''
    if use_fp16:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float16)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float16)
    else:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[:, c] = torch.clamp(image_tensor[:, c], -m / s, (1 - m) / s)
    return image_tensor


def denormalize(image_tensor, use_fp16=False):
    '''
    convert floats back to input
    '''
    if use_fp16:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float16)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float16)
    else:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[:, c] = torch.clamp(image_tensor[:, c] * s + m, 0, 1)

    return image_tensor


class EMA(object):
    def __init__(self, alpha, initial_value=None):
        self.alpha = alpha
        self.value = initial_value

    @torch.no_grad()
    def ema_update(self, x):
        if self.value is None:
            self.value = x
        else:
            self.value = self.alpha * self.value + (1 - self.alpha) * x


class BNFeatureHook():
    def __init__(self, module, training_momentum=0.4, flatness_weight=0.25):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.dd_var = 0.
        self.dd_mean = 0.
        self.momentum = training_momentum
        self.bn_statics_list = []
        self.ema_tag = False
        self.flatness = False
        self.closeness = False
        self.flatness_weight = flatness_weight

    def set_ori(self, flatness=False):
        self.ema_tag = False
        self.flatness = flatness
    
    def set_ema(self):
        self.ema_tag = True

    def hook_fn(self, module, input, output):
        nch = input[0].shape[1]
        input_0 = input[0]
        mean = input_0.mean([0, 2, 3])
        var = (input_0.permute(1, 0, 2, 3).contiguous().reshape([nch, -1])).var(1, unbiased=False)

        if not self.ema_tag:
            with torch.no_grad():
                if isinstance(self.dd_var, int):
                    self.dd_var = var
                    self.dd_mean = mean
                else:
                    self.dd_var = self.momentum * self.dd_var + (1 - self.momentum) * var
                    self.dd_mean = self.momentum * self.dd_mean + (1 - self.momentum) * mean
            r_feature = torch.norm(module.running_var.data - (self.dd_var + var - var.detach()), 2) + \
                        torch.norm(module.running_mean.data - (self.dd_mean + mean - mean.detach()), 2)
            if self.flatness:
                r_feature = r_feature + self.flatness_weight * (torch.norm(self.bn_statics_list[0] - (self.dd_var + var - var.detach()), 2) + \
                            torch.norm(self.bn_statics_list[1] - (self.dd_mean + mean - mean.detach()), 2))
            self.r_feature = r_feature
        else:
            self.bn_statics_list = [var, mean]

    def close(self):
        self.hook.remove()


class ConvFeatureHook():
    def __init__(self, module=None, save_path="./", data_number=1281167, name=None, gpu=0, training_momentum=0.4,
                 drop_rate=0.4, flatness_weight=0.25):

        self.module = module
        if module is not None and name is not None:
            self.hook = module.register_forward_hook(self.post_hook_fn)
        else:
            raise ModuleNotFoundError("module and name can not be None!")
        self.data_number = data_number
        self.dd_var = 0.
        self.dd_mean = 0.
        self.patch_var = 0.
        self.patch_mean = 0.
        self.flatness_weight = flatness_weight
        self.momentum = training_momentum  # origin = 0.2
        self.drop_rate = drop_rate  # 0.0 0.4 0.8
        dir = os.path.join(save_path, "ConvFeatureHook", name)
        if not os.path.exists(dir):
            os.makedirs(dir, exist_ok=True)
        self.save_path = os.path.join(save_path, "ConvFeatureHook", name, "running.npz")
        if os.path.exists(self.save_path):
            npz_file = np.load(self.save_path)
            self.load_tag = True
            self.running_dd_var = torch.from_numpy(npz_file["running_dd_var"]).cuda(gpu)
            self.running_dd_mean = torch.from_numpy(npz_file["running_dd_mean"]).cuda(gpu)
            self.running_patch_var = torch.from_numpy(npz_file["running_patch_var"]).cuda(gpu)
            self.running_patch_mean = torch.from_numpy(npz_file["running_patch_mean"]).cuda(gpu)
        else:
            self.load_tag = False
            self.running_dd_var = 0.
            self.running_dd_mean = 0.
            self.running_patch_var = 0.
            self.running_patch_mean = 0.
        self.conv_statics_list = []
        self.ema_tag = False
        self.flatness = False

    def set_ori(self, flatness=False):
        self.ema_tag = False
        self.flatness = flatness
    
    def set_ema(self):
        self.ema_tag = True

    def save(self):
        npz_file = {"running_dd_var": self.running_dd_var.cpu().numpy() if isinstance(self.running_dd_var,
                                                                                      torch.Tensor) else self.running_dd_var,
                    "running_dd_mean": self.running_dd_mean.cpu().numpy() if isinstance(self.running_dd_mean,
                                                                                        torch.Tensor) else self.running_dd_mean,
                    "running_patch_var": self.running_patch_var.cpu().numpy() if isinstance(self.running_patch_var,
                                                                                            torch.Tensor) else self.running_patch_var,
                    "running_patch_mean": self.running_patch_mean.cpu().numpy() if isinstance(self.running_patch_mean,
                                                                                              torch.Tensor) else self.running_patch_mean}
        print(npz_file)
        np.savez(self.save_path, **npz_file)

    def set_hook(self, pre=True):
        if hasattr(self, "hook"):
            self.close()
        if pre:
            self.hook = self.module.register_forward_hook(self.pre_hook_fn)
        else:
            self.hook = self.module.register_forward_hook(self.post_hook_fn)

    @torch.no_grad()
    def pre_hook_fn(self, module, input, output):
        nch = input[0].shape[1]
        bs = input[0].shape[0]
        input_0 = input[0]
        dd_mean = input_0.mean([0, 2, 3])
        dd_var = (input_0.permute(1, 0, 2, 3).contiguous().reshape([nch, -1])).var(1, unbiased=False)
        new_h, new_w = div_sixteen_mul(input_0.shape[2]), div_sixteen_mul(input_0.shape[3])
        new_input_0 = F.interpolate(input_0, [new_h, new_w], mode="bilinear")
        new_input_0 = einops.rearrange(new_input_0, "b c (u h) (v w) -> (u v) (b c h w)", h=16, w=16).contiguous()
        patch_mean = new_input_0.mean([1])
        patch_var = new_input_0.var([1], unbiased=False)
        self.running_dd_var += (dd_var * bs / self.data_number)
        self.running_dd_mean += (dd_mean * bs / self.data_number)
        self.running_patch_var += (patch_var * bs / self.data_number)
        self.running_patch_mean += (patch_mean * bs / self.data_number)

    def post_hook_fn(self, module, input, output):
        if random.random() > (1. - self.drop_rate):
            self.r_feature = torch.Tensor([0.]).to(input[0].device)
            return
        nch = input[0].shape[1]
        input_0 = input[0]
        dd_mean = input_0.mean([0, 2, 3])
        dd_var = (input_0.permute(1, 0, 2, 3).contiguous().reshape([nch, -1])).var(1, unbiased=False)
        new_h, new_w = div_sixteen_mul(input_0.shape[2]), div_sixteen_mul(input_0.shape[3])
        new_input_0 = F.interpolate(input_0, [new_h, new_w], mode="bilinear")
        new_input_0 = einops.rearrange(new_input_0, "b c (u h) (v w) -> (u v) (b c h w)", h=16, w=16).contiguous()
        patch_mean = new_input_0.mean([1])
        patch_var = new_input_0.var([1], unbiased=False)

        if not self.ema_tag:
            with torch.no_grad():
                if isinstance(self.dd_var, int):
                    self.dd_var = dd_var
                    self.dd_mean = dd_mean
                    self.patch_var = patch_var
                    self.patch_mean = patch_mean
                else:
                    self.dd_var = self.momentum * self.dd_var + (1 - self.momentum) * dd_var
                    self.dd_mean = self.momentum * self.dd_mean + (1 - self.momentum) * dd_mean
                    self.patch_var = self.momentum * self.patch_var + (1 - self.momentum) * patch_var
                    self.patch_mean = self.momentum * self.patch_mean + (1 - self.momentum) * patch_mean

            r_feature = torch.norm(self.running_dd_var - (self.dd_var + dd_var - dd_var.detach()), 2) + \
                        torch.norm(self.running_dd_mean - (self.dd_mean + dd_mean - dd_mean.detach()), 2) + \
                        torch.norm(self.running_patch_mean - (self.patch_mean + patch_mean - patch_mean.detach()), 2) + \
                        torch.norm(self.running_patch_var - (self.patch_var + patch_var - patch_var.detach()), 2)
            if self.flatness:
                r_feature = r_feature + self.flatness_weight * (torch.norm(self.conv_statics_list[0] - (self.dd_var + dd_var - dd_var.detach()), 2) + \
                            torch.norm(self.conv_statics_list[1] - (self.dd_mean + dd_mean - dd_mean.detach()), 2) + \
                            torch.norm(self.conv_statics_list[2] - (self.patch_mean + patch_mean - patch_mean.detach()), 2) + \
                            torch.norm(self.conv_statics_list[3] - (self.patch_var + patch_var - patch_var.detach()), 2))

            self.r_feature = r_feature
        else:
            self.conv_statics_list = [dd_var, dd_mean, patch_mean, patch_var]

    def close(self):
        self.hook.remove()


class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out


def get_image_prior_losses(inputs_jit):
    diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
    diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
    diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
    diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]

    loss_var_l2 = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
    loss_var_l1 = (diff1.abs() / 255.0).mean() + (diff2.abs() / 255.0).mean() + (
            diff3.abs() / 255.0).mean() + (diff4.abs() / 255.0).mean()
    loss_var_l1 = loss_var_l1 * 255.0

    return loss_var_l1, loss_var_l2


from torchvision.datasets import ImageFolder


class PreImgPathCache(ImageFolder):
    def __init__(
            self,
            root,
            transforms,
    ):
        super(PreImgPathCache, self).__init__(root,transform=transforms)
        self.label2img = [[] for _ in range(len(self.classes))]
        for k, v in self.imgs:
            self.label2img[v].append(k)

    def random_img_sample(self,idx):
        imgpaths = self.label2img[idx]
        new_idx = np.random.choice(len(imgpaths),(1,),replace=False)[0]
        imgpath = imgpaths[new_idx]
        sample = self.loader(imgpath)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample