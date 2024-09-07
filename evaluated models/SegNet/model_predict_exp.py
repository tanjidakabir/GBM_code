# +
import argparse
import os
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import nibabel as nib
from model_utility_segnet import Img_Dataset_test, dice_return, SegNet2D
from collections import OrderedDict

# Load dataset from yml file
with open('./path_predict.yaml', 'r') as config_file:
    data_path = yaml.safe_load(config_file)

in_channels = 4
batch_size = 4
class_nu = 4
device_ids = [5, 6, 7, 8]

test_df = pd.read_csv(data_path['MH_test'])
MH_model_folder = data_path['MH_model_folder']
MH_predict_folder = data_path['MH_test_predict']

model_file = 'model_epoch_42.pth'

predict_folder = MH_predict_folder.split('/')[-2]
isExist = os.path.exists(MH_predict_folder + predict_folder)
if not isExist:
    os.makedirs(MH_predict_folder + predict_folder)

test_data = Img_Dataset_test(test_df)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=1)

eval_dice = []
pd_list = []

def remove_module_prefix(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k  # remove 'module.' prefix
        new_state_dict[name] = v
    return new_state_dict

model = SegNet2D(in_chn=in_channels, out_chn=class_nu)
state_dict = torch.load(MH_model_folder + model_file, map_location="cpu")
state_dict = remove_module_prefix(state_dict)
model.load_state_dict(state_dict)

model = nn.DataParallel(model, device_ids=device_ids)
model = model.cuda(device_ids[0])  # Ensure model is moved to the first device
model.eval()

for count, batch in tqdm(enumerate(test_loader), desc="Predicting"):
    images, seg, pt_id = batch
    pd_list.append(pt_id[0])  # Assuming pt_id is a list of a single item
    seg_results = []

    with torch.no_grad():
        # Get the batch size, channels, height, width, and depth
        batch_size, channels, height, width, depth = images.size()

        # Reshape the images for 2D processing by the model
        images = images.permute(0, 4, 1, 2, 3).reshape(-1, in_channels, height, width)
        images = images.cuda(device_ids[0])  # Move images to the same device as the model

        # Perform the model prediction
        output = model(images)

        # Reshape the output back to the original 3D shape
        output = output.view(batch_size, depth, class_nu, height, width).permute(0, 2, 3, 4, 1).cpu()
        seg_results.append(output)

    # Stack the results along a new dimension
    seg_results = torch.stack(seg_results)

    # Remove any singleton dimensions
    seg_results = torch.squeeze(seg_results)

    # Compute the average segmentation result across the stacked dimension
    seg_layer_avg = torch.mean(seg_results, dim=0)

    # Get the most probable class for each voxel
    _, indices = seg_layer_avg.max(0)
    indices = indices.cpu().detach().numpy().astype('uint8')

    # Save the segmentation result as a NIfTI image
    img = nib.Nifti1Image(indices, np.eye(4))
    save_path = MH_predict_folder + predict_folder + '/' + str(pt_id[0]) + '_segment_pred.nii.gz'
    nib.save(img, save_path)
    print('Saved example_segment_pred' + '_' + str(pt_id[0]))

    # Debug: Print shapes to identify mismatch
    print(f"Shape of indices: {indices.shape}")
    print(f"Shape of seg: {seg.shape}")

    # Compute the Dice scores for the segmentation
    seg_1 = seg.cpu().detach().numpy().astype('uint8')
    dice_scores = dice_return(indices.squeeze(), seg_1.squeeze())  # Ensure this returns 4 values
    eval_dice.append(dice_scores)

# Create a DataFrame with the Dice scores and patient IDs
eval_dice_df = pd.DataFrame(eval_dice, columns=['background_dice', 'edema_dice', 'enh_dice', 'nonenc_dice'])
eval_dice_df['pt_id'] = pd_list

# Save the Dice scores to a CSV file
eval_dice_df.to_csv(MH_predict_folder + predict_folder + '/' + 'all_dice.csv', index=False)

# -

output

indices
