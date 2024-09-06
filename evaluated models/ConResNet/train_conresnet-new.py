import argparse
import sys
sys.path.append("..")

import torch
import numpy as np
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

# +
import os
import os.path as osp
from models.ConResNet import ConResNet
#from dataset.BraTSDataSet import BraTSDataSet, BraTSValDataSet
from dataset.BraTSDataSet_2023 import BraTSDataSet, BraTSValDataSet

import timeit
from tensorboardX import SummaryWriter
from utils import loss
from utils.engine import Engine
from math import ceil
# -

start = timeit.default_timer()

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_arguments():
    """
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="ConResNet for 3D Medical Image Segmentation.")

    parser.add_argument("--data_dir", type=str, default='/home/BraTS23_dataset/')
    parser.add_argument("--train_list", type=str, default='BraTS23_dataset/list/train_list.txt')
    parser.add_argument("--val_list", type=str, default='BraTS23_dataset/list/val_list.txt')
    parser.add_argument("--snapshot_dir", type=str, default='snapshots/conresnet/')
    parser.add_argument("--reload_path", type=str, default='snapshots/conresnet/ConResNet_40000.pth')
    parser.add_argument("--reload_from_checkpoint", type=str2bool, default=False)
    parser.add_argument("--input_size", type=str, default='80,160,160')
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument('--local_rank', type=int, default=4)
    parser.add_argument("--num_steps", type=int, default=10000)
    parser.add_argument("--start_iters", type=int, default=0)
    parser.add_argument("--val_pred_every", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--weight_std", type=str2bool, default=True)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--power", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--ignore_label", type=int, default=255)
    parser.add_argument("--is_training", action="store_true")
    parser.add_argument("--random_mirror", type=str2bool, default=False)
    parser.add_argument("--random_scale", type=str2bool, default=False)
    parser.add_argument("--random_seed", type=int, default=1234)

    return parser


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))

def adjust_learning_rate(optimizer, i_iter, lr, num_steps, power):
    lr = lr_poly(lr, i_iter, num_steps, power)
    optimizer.param_groups[0]['lr'] = lr
    return lr


def dice_score(preds, labels):
    assert preds.shape[0] == labels.shape[0], "predict & target batch size don't match"
    predict = preds.contiguous().view(preds.shape[0], -1)
    target = labels.contiguous().view(labels.shape[0], -1)

    num = torch.sum(torch.mul(predict, target), dim=1)
    den = torch.sum(predict, dim=1) + torch.sum(target, dim=1) + 1

    dice = 2*num / den

    return dice.mean()


def compute_dice_score(preds, labels):

    preds = F.sigmoid(preds)

    pred_ET = preds[:, 0, :, :, :]
    pred_WT = preds[:, 1, :, :, :]
    pred_TC = preds[:, 2, :, :, :]
    label_ET = labels[:, 0, :, :, :]
    label_WT = labels[:, 1, :, :, :]
    label_TC = labels[:, 2, :, :, :]
    dice_ET = dice_score(pred_ET, label_ET).cpu().data.numpy()
    dice_WT = dice_score(pred_WT, label_WT).cpu().data.numpy()
    dice_TC = dice_score(pred_TC, label_TC).cpu().data.numpy()
    return dice_ET, dice_WT, dice_TC


def predict_sliding(net, imagelist, tile_size, classes):
    image, image_res = imagelist
    image_size = image.shape
    overlap = 1 / 3

    strideHW = ceil(tile_size[1] * (1 - overlap))
    strideD = ceil(tile_size[0] * (1 - overlap))
    tile_deps = int(ceil((image_size[2] - tile_size[0]) / strideD) + 1)
    tile_rows = int(ceil((image_size[3] - tile_size[1]) / strideHW) + 1)
    tile_cols = int(ceil((image_size[4] - tile_size[2]) / strideHW) + 1)
    full_probs = np.zeros((image_size[0], classes, image_size[2], image_size[3], image_size[4])).astype(np.float32)
    count_predictions = np.zeros((image_size[0], classes, image_size[2], image_size[3], image_size[4])).astype(np.float32)
    full_probs = torch.from_numpy(full_probs).cuda(4)
    count_predictions = torch.from_numpy(count_predictions).cuda(4)

    for dep in range(tile_deps):
        for row in range(tile_rows):
            for col in range(tile_cols):
                d1 = int(dep * strideD)
                x1 = int(col * strideHW)
                y1 = int(row * strideHW)
                d2 = min(d1 + tile_size[0], image_size[2])
                x2 = min(x1 + tile_size[2], image_size[4])
                y2 = min(y1 + tile_size[1], image_size[3])
                d1 = max(int(d2 - tile_size[0]), 0)
                x1 = max(int(x2 - tile_size[2]), 0)
                y1 = max(int(y2 - tile_size[1]), 0)

                img = image[:, :, d1:d2, y1:y2, x1:x2]
                img_res = image_res[:, :, d1:d2, y1:y2, x1:x2]

                prediction = net([img, img_res])
                prediction = prediction[0]

                count_predictions[:, :, d1:d2, y1:y2, x1:x2] += 1
                full_probs[:, :, d1:d2, y1:y2, x1:x2] += prediction

    full_probs /= count_predictions
    return full_probs


def validate(input_size, model, ValLoader, num_classes):
    # start to validate
    val_ET = 0.0
    val_WT = 0.0
    val_TC = 0.0

    for index, batch in enumerate(ValLoader):
        #print('%d processd'%(index))
        image, image_res, label, size, name, affine = batch
        image = image.cuda(4)
        image_res = image_res.cuda(4)
        label = label.cuda(4)
        with torch.no_grad():
            pred = predict_sliding(model, [image, image_res], input_size, num_classes)
            dice_ET, dice_WT, dice_TC = compute_dice_score(pred, label)
            val_ET += dice_ET
            val_WT += dice_WT
            val_TC += dice_TC

    return val_ET/(index+1), val_WT/(index+1), val_TC/(index+1)

def get_train_loader(batch_size, train_dataset):
    train_sampler = None
    is_shuffle = True
    batch_size = batch_size

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                       batch_size=batch_size,
                                       num_workers=2,
                                       drop_last=False,
                                       shuffle=is_shuffle,
                                       pin_memory=True,
                                       sampler=train_sampler)

    return train_loader, train_sampler


def get_test_loader( test_dataset):
    test_sampler = None
    is_shuffle = False


       
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                       batch_size=4,
                                       num_workers=2,
                                       drop_last=False,
                                       shuffle=is_shuffle,
                                       pin_memory=True,
                                       sampler=test_sampler)

    return test_loader, test_sampler


# +
def main():
    """Create the ConResNet model and then start the training."""
    parser = get_arguments()
    args = parser.parse_args()
    writer = SummaryWriter(args.snapshot_dir)
    d, h, w = map(int, args.input_size.split(','))
    input_size = (d, h, w)
    cudnn.benchmark = True
    seed = args.random_seed
    model = ConResNet(input_size, num_classes=args.num_classes, weight_std=True)
    model.train()
    model=torch.nn.DataParallel(model, device_ids=[4,5,12])
    model.to(f'cuda:{model.device_ids[0]}')
    optimizer = optim.Adam(
        [{'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr': args.learning_rate}],
        lr=args.learning_rate, weight_decay=args.weight_decay)
    
    loss_D = loss.DiceLoss4BraTS().to(4)
    loss_BCE = loss.BCELoss4BraTS().to(4)

    loss_B = loss.BCELossBoud().to(4)

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    trainloader, train_sampler = get_train_loader(args.batch_size,BraTSDataSet(args.data_dir, args.train_list, max_iters=args.num_steps * args.batch_size, crop_size=input_size,
                        scale=args.random_scale, mirror=args.random_mirror))
    valloader, val_sampler = get_test_loader(BraTSValDataSet(args.data_dir, args.val_list))
    for i_iter, batch in enumerate(trainloader):
        i_iter += args.start_iters
        images, images_res, labels, labels_res = batch
        images = images#.cuda()
        images_res = images_res#.cuda()
        labels = labels.cuda(4)
        labels_res = labels_res.cuda(4)

        optimizer.zero_grad()
        lr = adjust_learning_rate(optimizer, i_iter, args.learning_rate, args.num_steps, args.power)

        preds= model([images, images_res])
        preds=preds
        preds_seg = preds[0].cuda(4)
        preds_res = preds[1].cuda(4)
        preds_resx2 = preds[2].cuda(4)
        preds_resx4 = preds[3].cuda(4)

        term_seg_Dice = loss_D.forward(preds_seg, labels)
        term_seg_BCE = loss_BCE.forward(preds_seg, labels)

        term_res_BCE = loss_B.forward(preds_res, labels_res)
        term_resx2_BCE = loss_B.forward(preds_resx2, labels_res)
        term_resx4_BCE = loss_B.forward(preds_resx4, labels_res)

        term_all = term_seg_Dice + term_seg_BCE + term_res_BCE + 0.5 * (term_resx2_BCE +term_resx4_BCE)
        term_all.backward()

        optimizer.step()

        if i_iter % 100 == 0:# and (args.local_rank == 0):
            writer.add_scalar('learning_rate', lr, i_iter)
            writer.add_scalar('loss', term_all.cpu().data.numpy(), i_iter)

        print('iter = {} of {} completed, lr = {:.4}, seg_loss = {:.4}, res_loss = {:.4}'.format(
                i_iter, args.num_steps, lr, (term_seg_Dice+term_seg_BCE).cpu().data.numpy(), (term_res_BCE+term_resx2_BCE+term_resx4_BCE).cpu().data.numpy()))


        if i_iter >= args.num_steps - 1:# and (args.local_rank == 0):
            print('save last model ...')
            torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'ConResNet_' + str(args.num_steps) + '.pth'))
            break

        if i_iter % args.val_pred_every == 0 and i_iter!=0:# and (args.local_rank == 0):
            print('save model ...')
            torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'ConResNet_' + str(i_iter) + '.pth'))

            # val
        if i_iter % args.val_pred_every == 0:
            print('validate ...')
            val_ET, val_WT, val_TC = validate(input_size, model, valloader, args.num_classes)
            writer.add_scalar('Val_ET_Dice', val_ET, i_iter)
            writer.add_scalar('Val_WT_Dice', val_WT, i_iter)
            writer.add_scalar('Val_TC_Dice', val_TC, i_iter)
            print('Validate iter = {}, ET = {:.2}, WT = {:.2}, TC = {:.2}'.format(i_iter, val_ET, val_WT, val_TC))

end = timeit.default_timer()
print(end - start, 'seconds')

# -

if __name__ == '__main__':
    main()



