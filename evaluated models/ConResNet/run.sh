#!/bin/bash


CUDA_VISIBLE_DEVICES=2,3 nohup python -m torch.distributed.launch --nproc_per_node=2 --master_port=$RANDOM train_conresnet.py \
--data_dir='/home/MICCAI_old/MICCAI_BraTS2020_kanglin/' \
--train_list='/home/MICCAI_old/MICCAI_BraTS2020_kanglin/list/brats_2020/train_list.txt' \
--val_list='/home/MICCAI_old/MICCAI_BraTS2020_kanglin/list/brats_2020/val_list.txt' \
--snapshot_dir='./save_model/' \
--input_size='80,160,160' \
--batch_size=10 \
--num_gpus=4 \
--num_steps=40000 \
--val_pred_every=2000 \
--learning_rate=1e-4 \
--num_classes=3 \
--num_workers=4 \
--random_mirror=True \
--random_scale=True \
> path-to-save-log-file/log.file 2>&1 &
