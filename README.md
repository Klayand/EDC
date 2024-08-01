# EDC

Offical Implementation of our research:

> **Elucidating the Design Space of Dataset Condensation ** <br>


## Get Started

### Prerequisites
Our code can be easily run, you only need install following packages and dependency:
```bash
# create the conda environment
conda create <name env> -file env.yaml

# install the dependency and packages
pip install -r requirements.txt
```

### File Directories
For `CIFAR-10`, `CIFAR-100`, `ImageNet_10` and `Tiny_ImageNet` datasets, Each of them consists of four parts:
```dotenv
├─Branch_CIFAR_100
│  ├─recover   # Get the synthetic images
│  │  └─models
│  ├─relabel   # Relabel the sythetic images
│  │  └─models
│  ├─squeeze   # Collect the statistics
│  │  └─models
│  └─train     # Evaluate the performance of the synthetic dataset
```
For `ImageNet-1K` dataset, it consists of three parts:
```dotenv
├─Branch_full_ImageNet_1k
│  ├─recover  # Get the synthetic images
│  │  └─statistic  # Statistics collected from squeeze stage
│  │      ├─BNFeatureHook
│  │      └─ConvFeatureHook
│  ├─relabel  # Relabel the sythetic images
│  └─train    # Evaluate the performance of the synthetic dataset
```

### Usage Example
For `CIFAR-100`(same with `CIFAR-10`, `Tiny-ImageNet` and `ImageNet_10`), you first need to run `squeeze.sh` in `squeeze` folder, for `ImageNet-1k`, use the released statistics file:
```bash
bash squeeze.sh
```
Inside this file, you can choose the models which you want to collect the statistics from:
```bash
python train.py --model ResNet18 --dataset CIFAR-100 --data_path /path to cifar-100  --squeeze_path /path to collected statistics

python train.py --model MobileNetV2 --dataset CIFAR-100 --data_path /path to cifar-100  --squeeze_path /path to collected statistics

python train.py --model ShuffleNetV2_0_5 --dataset CIFAR-100 --data_path /path to cifar-100  --squeeze_path /path to collected statistics

python train.py --model WRN_16_2 --dataset CIFAR-100 --data_path /path to cifar-100  --squeeze_path /path to collected statistics
 
python train.py --model ConvNetW128 --dataset CIFAR-100 --data_path /path to cifar-100  --squeeze_path /path to collected statistics
```
*Reminder: You can run the `squeeze.sh` to get the statistics, or download from the [link](https://github.com/Klayand/EDC/releases/tag/v1.0) directly.*

Then, you need to execute the second stage `recover` to get the synthetic data, for `CIFAR-10` and `CIFAR-100`, you don`t need to get the initialized images, you can directly run:
```bash
CUDA_VISIBLE_DEVICES=0,1 python data_synthesis_with_svd_with_db_with_all_statistic.py \
    --arch-name "resnet18" \
    --exp-name "EDC_CIFAR_100_Recover_IPC_10" \
    --batch-size 100 \
    --lr 0.05 \
    --ipc-number 10 \
    --iteration 4000 \
    --train-data-path /path to cifar-100 \
    --l2-scale 0 --tv-l2 0 --r-loss 0.01 --nuc-norm 1. \
    --verifier --store-best-images --gpu-id 0,1
```
However, for `ImageNet-1k`, `Tiny-ImageNet` and `ImageNet-10`, you need first to get the initialized images for RDED, then recover:
```bash
# first to run this command to get the RDED initialization images.
CUDA_VISIBLE_DEVICES=1,2 python data_synthesis_without_optim.py \
    --exp-name "WO_OPTIM_ImageNet_1k_Recover_IPC_10" \
    --ipc-number 10 \
    --train-data-path /path to ImageNet train --gpu-id 1,2
    
# then use the initialized images to get the synthetic data.
 CUDA_VISIBLE_DEVICES=5,6 python recover.py \
     --arch-name "resnet18" \
     --exp-name "EDC_ImageNet_1k_Recover_IPC_10" \
     --batch-size 80 \
     --lr 0.05 --category-aware "global" \
     --ipc-number 10 --training-momentum 0.8  \
     --iteration 1000 --drop-rate 0.0 \
     --train-data-path /path to ImageNet train \
     --l2-scale 0 --tv-l2 0 --r-loss 0.1 --nuc-norm 1. \
     --verifier --store-best-images --gpu-id 5,6 --initial-img-dir /path to the initialized images \
     --statistic-path /path to collected statistics
```
Then run the `relabel.sh`:
```bash
CUDA_VISIBLE_DEVICES=0 python generate_soft_label_with_db.py \
    -b 100 \
    -j 8 \
    --epochs 300 \
    --fkd-seed 42 \
    --input-size 224 \
    --min-scale-crops 0.5 \
    --max-scale-crops 1 \
    --use-fp16 --candidate-number 4 \
    --fkd-path /path to store the synthetic label \
    --mode 'fkd_save' \
    --mix-type 'cutmix' \
    --data /path to synthetic data
```
*Reminder:  the batch-size `--batch-size` in `recover.sh` should be the same with the batch-size `-b` in `relabel.sh`*

After that, you successfully get the synthetic dataset, then you can evaluate it:
```bash
CUDA_VISIBLE_DEVICES=0 python train_FKD_parallel.py \
    --wandb-project 'final_efficientnet_b0_fkd' \
    --batch-size 100 \
    --model "efficientnet_b0" \
    --ls-type cos --loss-type "mse_gt" --ce-weight 0.025 \
    -j 4 --gradient-accumulation-steps 1  --st 2 --ema-dr 0.99 \
    -T 20 --gpu-id 0 \
    --mix-type 'cutmix' \
    --output-dir ./save/final_efficientnet_b0_fkd/ \
    --train-dir /path to the synthetic image \
    --val-dir /data/imagenet1k/val/ \
    --fkd-path /path to the synthetic label
```

This is all about the whole process.
For `ImageNet-1k`, recommend you to directly download the collected statistics for simplicity. **[Statistics Download Links](https://github.com/Klayand/EDC/releases/download/v1.0/statistic_imagenet1k.zip)**








