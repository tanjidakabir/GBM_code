# +
import argparse
from model_utility_segnet import *

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import yaml
from monai.losses import GeneralizedDiceFocalLoss
from torch.optim.lr_scheduler import StepLR
from torch.cuda.amp import GradScaler, autocast


def train_model(args):
    with open(args['config_file'], 'r') as config_file:
        data_path = yaml.safe_load(config_file)

    print("Start load dataset")
    data_df = pd.read_csv(data_path['train_data'])
    data_df['index'] = range(len(data_df))
    train_df = data_df.sample(frac=0.9)
    val_df = data_df[~data_df['index'].isin(train_df['index'])]
    print("build dataset")
    train_data = Img_Dataset(train_df)
    train_loader = DataLoader(train_data, batch_size=args['batch_size'], shuffle=True, num_workers=2)
    val_data = Img_Dataset(val_df)
    val_loader = DataLoader(val_data, batch_size=args['batch_size'], shuffle=True, num_workers=2)
    print("building model")

    model = SegNet2D(in_chn=args['input_channels'])  # Ensure input_channels is set correctly
    model = nn.DataParallel(model, device_ids=args['device_ids'])
    torch.cuda.set_device(args['device_ids'][0])
    model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args['learning_rate'], weight_decay=1e-5)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    criterion = GeneralizedDiceFocalLoss(softmax=True)
    scaler = GradScaler()  # Initialize the GradScaler for mixed precision training

    if not os.path.exists(args['model_dir'] + args['model_prefix']):
        os.makedirs(args['model_dir'] + args['model_prefix'])

    all_train_loss = []
    all_val_loss = []
    prev_val_loss = float('inf')

    epoch_progress = tqdm(range(1, args['epoch'] + 1), desc="Epochs")
    for epoch in epoch_progress:
        epoch_train_loss = 0

        model.train()
        iteration_progress = tqdm(train_loader, desc=f"Training Epoch {epoch}")
        for count, batch in enumerate(iteration_progress):
            images, seg = batch

            batch_size, channels, height, width, depth = images.size()
            images = images.permute(0, 4, 1, 2, 3).reshape(-1, channels, height, width)  # Reshape for 2D processing
            seg = seg.permute(0, 3, 1, 2).reshape(-1, height, width)  # Reshape for 2D processing

            seg = seg.unsqueeze(1)  # Add channel dimension to seg to make it 4D (B, 1, H, W)
            seg = expand_as_one_hot(seg.long(), args['class_nu'], ignore_index=None)
            seg = seg.squeeze(2)  # Remove the extra channel dimension
            images = images.cuda()
            seg = seg.cuda()

            optimizer.zero_grad()
            with autocast():  # Casts operations to mixed precision
                output = model(images)
                train_loss = criterion(output, seg)

            scaler.scale(train_loss).backward()  # Scales the loss for mixed precision
            scaler.step(optimizer)  # Unscales the gradients and calls the optimizer
            scaler.update()  # Updates the scale for next iteration

            epoch_train_loss += train_loss.item()
            iteration_progress.set_postfix(train_loss=train_loss.item())

        scheduler.step()
        epoch_train_loss /= len(train_loader)
        all_train_loss.append(epoch_train_loss)

        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            val_iteration_progress = tqdm(val_loader, desc=f"Validation Epoch {epoch}")
            for count, batch in enumerate(val_iteration_progress):
                images, seg = batch

                batch_size, channels, height, width, depth = images.size()
                images = images.permute(0, 4, 1, 2, 3).reshape(-1, channels, height, width)  # Reshape for 2D processing
                seg = seg.permute(0, 3, 1, 2).reshape(-1, height, width)  # Reshape for 2D processing

                seg = seg.unsqueeze(1)  # Add channel dimension to seg to make it 4D (B, 1, H, W)
                seg = expand_as_one_hot(seg.long(), args['class_nu'], ignore_index=None)
                seg = seg.squeeze(2)  # Remove the extra channel dimension
                images = images.cuda()
                seg = seg.cuda()

                with autocast():  # Casts operations to mixed precision
                    output = model(images)
                    val_loss = criterion(output, seg)
                epoch_val_loss += val_loss.item()
                val_iteration_progress.set_postfix(val_loss=val_loss.item())

        epoch_val_loss /= len(val_loader)
        all_val_loss.append(epoch_val_loss)

        if epoch_val_loss < prev_val_loss:
            torch.save(model.state_dict(), os.path.join(args['model_dir'], args['model_prefix'], 'model_epoch_{}.pth'.format(epoch)))
            prev_val_loss = epoch_val_loss

    loss_df = pd.DataFrame({'train_loss': all_train_loss, 'val_loss': all_val_loss, 'epoch': range(1, args['epoch'] + 1)})
    loss_df.to_csv(os.path.join(args['model_dir'], args['model_prefix'], 'loss.csv'), index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SegNet Model')

    parser.add_argument('--config_file', type=str, default='./path_2.yaml', help='Path to configuration file')
    parser.add_argument('--epoch', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.005, help='Learning rate')
    parser.add_argument('--class_nu', type=int, default=4, help='Number of output classes')
    parser.add_argument('--model_dir', type=str, default='./models/', help='Directory to save models')
    parser.add_argument('--model_prefix', type=str, default='SegNet', help='Prefix for saved models')
    parser.add_argument('--device_ids', type=int, nargs='+', default=[5,6,7,8], help='List of device IDs for multi-GPU training')
    parser.add_argument('--input_channels', type=int, default=4, help='Number of input channels for the model')

    args = parser.parse_args()
    args = vars(args)

    train_model(args)


