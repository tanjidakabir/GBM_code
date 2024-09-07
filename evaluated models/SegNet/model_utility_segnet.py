# +
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import pandas as pd
import numpy as np
from collections import OrderedDict
from monai.losses import GeneralizedDiceLoss, DiceCELoss, DiceFocalLoss, GeneralizedDiceFocalLoss

class Img_Dataset (Dataset):

    def __init__(self, dataset):
        
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        data_index = self.data.iloc[idx]
        images_path=data_index.images_path
        seg_path=data_index.seg_path
        with open(images_path, 'rb') as f:
            images = pickle.load(f)
        
        with open(seg_path, 'rb') as f:
            seg = pickle.load(f)       
        return images, seg

class Img_Dataset_test(Dataset):
    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_index = self.data.iloc[idx]
        images_path = data_index.images_path
        seg_path = data_index.seg_path

        with open(images_path, 'rb') as f:
            images = pickle.load(f)

        with open(seg_path, 'rb') as f:
            seg = pickle.load(f)

        return images, seg, data_index.pt_id

def expand_as_one_hot(input, C, ignore_index=None):
    assert input.dim() == 4

    # expand the input tensor to Nx1xDxHxW before scattering
    input = input.unsqueeze(1)
    # create result tensor shape (NxCxDxHxW)
    shape = list(input.size())
    shape[1] = C

    if ignore_index is not None:
        # create ignore_index mask for the result
        mask = input.expand(shape) == ignore_index
        # clone the src tensor and zero out ignore_index in the input
        input = input.clone()
        input[input == ignore_index] = 0
        # scatter to get the one-hot tensor
        result = torch.zeros(shape).to(input.device).scatter_(1, input, 1)
        # bring back the ignore_index in the result
        result[mask] = ignore_index
        return result
    else:
        # scatter to get the one-hot tensor
        return torch.zeros(shape).to(input.device).scatter_(1, input, 1)
    

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Apply softmax to the inputs
        inputs = F.softmax(inputs, dim=1)
        
        # Gather the probabilities of the target classes
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1)).permute(0, 3, 1, 2)
        pt = (inputs * targets_one_hot).sum(dim=1)  # pt is the probability of the true class

        # Compute the focal loss
        loss = -self.alpha * (1 - pt) ** self.gamma * torch.log(pt)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# def dice_coef2(y_true, y_pred):
#     y_true_f = y_true.flatten()
#     y_pred_f = y_pred.flatten()
#     union = np.sum(y_true_f) + np.sum(y_pred_f)
#     if union==0: return 1
    
#     intersection = np.sum(y_true_f * y_pred_f)
    
#     return 2. * intersection / union

# def dice_return(newseg,indices ):
#     background_dice=[]
#     enh_dice=[]
#     nonenc_dice=[]
#     edema_dice=[]

#     seg = newseg
#     pred = indices
    
#     newsegback=np.where(seg==0,1,0)
#     newpredback=np.where(pred==0,1,0)
#     bdice=dice_coef2(newsegback, newpredback)
#     background_dice.append(bdice)
    
#         #'edema'
#     newsegback=np.where(seg==1,1,0)
#     newpredback=np.where(pred==1,1,0)
#     bdice=dice_coef2(newsegback, newpredback)
#     edema_dice.append(bdice)
#         #'enhencing'
#     newsegback=np.where(seg==2,1,0)
#     newpredback=np.where(pred==2,1,0)
#     bdice=dice_coef2(newsegback, newpredback)
#     enh_dice.append(bdice)
#         #'nonenhencing'
#     newsegback=np.where(seg==3,1,0)
#     newpredback=np.where(pred==3,1,0)
#     bdice=dice_coef2(newsegback, newpredback)
#     nonenc_dice.append(bdice)
#     print(background_dice[0],edema_dice[0],enh_dice[0],nonenc_dice[0])
    
#     return [background_dice[0],edema_dice[0],enh_dice[0],nonenc_dice[0]]

def dice_coef2(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    union = np.sum(y_true_f) + np.sum(y_pred_f)
    if union == 0: 
        return 1
    
    intersection = np.sum(y_true_f * y_pred_f)
    return 2. * intersection / union

def dice_return(newseg, indices):
    background_dice = []
    enh_dice = []
    nonenc_dice = []
    edema_dice = []

    seg = newseg
    pred = indices

    for i in range(4):
        newseg_class = np.where(seg == i, 1, 0)
        newpred_class = np.where(pred == i, 1, 0)
        dice = dice_coef2(newseg_class, newpred_class)
        
        # Debug: Print intermediate arrays for one sample
        if i == 0:
            print(f"Background - seg_class sum: {np.sum(newseg_class)}, pred_class sum: {np.sum(newpred_class)}")
        elif i == 1:
            print(f"Edema - seg_class sum: {np.sum(newseg_class)}, pred_class sum: {np.sum(newpred_class)}")
        elif i == 2:
            print(f"Enhancing - seg_class sum: {np.sum(newseg_class)}, pred_class sum: {np.sum(newpred_class)}")
        elif i == 3:
            print(f"Non-enhancing - seg_class sum: {np.sum(newseg_class)}, pred_class sum: {np.sum(newpred_class)}")

        if i == 0:
            background_dice.append(dice)
        elif i == 1:
            edema_dice.append(dice)
        elif i == 2:
            enh_dice.append(dice)
        elif i == 3:
            nonenc_dice.append(dice)

    return [background_dice[0], edema_dice[0], enh_dice[0], nonenc_dice[0]]


class SegNet2D(nn.Module):
    def __init__(self, in_chn=4, out_chn=4, BN_momentum=0.5):
        super(SegNet2D, self).__init__()

        self.in_chn = in_chn
        self.out_chn = out_chn

        self.MaxEn = nn.MaxPool2d(2, stride=2, return_indices=True)
        self.MaxDe = nn.MaxUnpool2d(2, stride=2)

        self.ConvEn11 = nn.Conv2d(self.in_chn, 64, kernel_size=3, padding=1)
        self.BNEn11 = nn.BatchNorm2d(64, momentum=BN_momentum)
        self.ConvEn12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.BNEn12 = nn.BatchNorm2d(64, momentum=BN_momentum)

        self.ConvEn21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.BNEn21 = nn.BatchNorm2d(128, momentum=BN_momentum)
        self.ConvEn22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.BNEn22 = nn.BatchNorm2d(128, momentum=BN_momentum)

        self.ConvEn31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.BNEn31 = nn.BatchNorm2d(256, momentum=BN_momentum)
        self.ConvEn32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.BNEn32 = nn.BatchNorm2d(256, momentum=BN_momentum)
        self.ConvEn33 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.BNEn33 = nn.BatchNorm2d(256, momentum=BN_momentum)

        self.ConvEn41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.BNEn41 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvEn42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNEn42 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvEn43 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNEn43 = nn.BatchNorm2d(512, momentum=BN_momentum)

        self.ConvEn51 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNEn51 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvEn52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNEn52 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvEn53 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNEn53 = nn.BatchNorm2d(512, momentum=BN_momentum)

        self.ConvDe53 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNDe53 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvDe52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNDe52 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvDe51 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNDe51 = nn.BatchNorm2d(512, momentum=BN_momentum)

        self.ConvDe43 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNDe43 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvDe42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNDe42 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvDe41 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.BNDe41 = nn.BatchNorm2d(256, momentum=BN_momentum)

        self.ConvDe33 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.BNDe33 = nn.BatchNorm2d(256, momentum=BN_momentum)
        self.ConvDe32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.BNDe32 = nn.BatchNorm2d(256, momentum=BN_momentum)
        self.ConvDe31 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.BNDe31 = nn.BatchNorm2d(128, momentum=BN_momentum)

        self.ConvDe22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.BNDe22 = nn.BatchNorm2d(128, momentum=BN_momentum)
        self.ConvDe21 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.BNDe21 = nn.BatchNorm2d(64, momentum=BN_momentum)

        self.ConvDe12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.BNDe12 = nn.BatchNorm2d(64, momentum=BN_momentum)
        self.ConvDe11 = nn.Conv2d(64, self.out_chn, kernel_size=3, padding=1)
        self.BNDe11 = nn.BatchNorm2d(self.out_chn, momentum=BN_momentum)

    def forward(self, x):
        # Encode
        x = F.relu(self.BNEn11(self.ConvEn11(x)))
        x = F.relu(self.BNEn12(self.ConvEn12(x)))
        x, ind1 = self.MaxEn(x)
        size1 = x.size()

        x = F.relu(self.BNEn21(self.ConvEn21(x)))
        x = F.relu(self.BNEn22(self.ConvEn22(x)))
        x, ind2 = self.MaxEn(x)
        size2 = x.size()

        x = F.relu(self.BNEn31(self.ConvEn31(x)))
        x = F.relu(self.BNEn32(self.ConvEn32(x)))
        x = F.relu(self.BNEn33(self.ConvEn33(x)))
        x, ind3 = self.MaxEn(x)
        size3 = x.size()

        x = F.relu(self.BNEn41(self.ConvEn41(x)))
        x = F.relu(self.BNEn42(self.ConvEn42(x)))
        x = F.relu(self.BNEn43(self.ConvEn43(x)))
        x, ind4 = self.MaxEn(x)
        size4 = x.size()

        x = F.relu(self.BNEn51(self.ConvEn51(x)))
        x = F.relu(self.BNEn52(self.ConvEn52(x)))
        x = F.relu(self.BNEn53(self.ConvEn53(x)))
        x, ind5 = self.MaxEn(x)
        size5 = x.size()

        # Decode
        x = self.MaxDe(x, ind5, output_size=size4)
        x = F.relu(self.BNDe53(self.ConvDe53(x)))
        x = F.relu(self.BNDe52(self.ConvDe52(x)))
        x = F.relu(self.BNDe51(self.ConvDe51(x)))

        x = self.MaxDe(x, ind4, output_size=size3)
        x = F.relu(self.BNDe43(self.ConvDe43(x)))
        x = F.relu(self.BNDe42(self.ConvDe42(x)))
        x = F.relu(self.BNDe41(self.ConvDe41(x)))

        x = self.MaxDe(x, ind3, output_size=size2)
        x = F.relu(self.BNDe33(self.ConvDe33(x)))
        x = F.relu(self.BNDe32(self.ConvDe32(x)))
        x = F.relu(self.BNDe31(self.ConvDe31(x)))

        x = self.MaxDe(x, ind2, output_size=size1)
        x = F.relu(self.BNDe22(self.ConvDe22(x)))
        x = F.relu(self.BNDe21(self.ConvDe21(x)))

        x = self.MaxDe(x, ind1, output_size=(size1[0], 64, size1[2]*2, size1[3]*2))
        x = F.relu(self.BNDe12(self.ConvDe12(x)))
        x = self.ConvDe11(x)

#         x = F.softmax(x, dim=1)
        return x

