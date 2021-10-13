import torch
from torch import multiprocessing, cuda
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.backends import cudnn

import numpy as np
import importlib
import argparse
import os
from numpy.linalg import lstsq
from scipy.linalg import orth
import voc12.dataloader
from misc import torchutils, imutils
import cv2
import coco14.dataloader



cudnn.enabled = True

parser = argparse.ArgumentParser()

# Environment
parser.add_argument("--num_workers", default=os.cpu_count()//2, type=int)
parser.add_argument("--voc12_root", default='Dataset/coco_2014', type=str,
                    help="Path to VOC 2012 Devkit, must contain ./JPEGImages as subdirectory.")
# Dataset
parser.add_argument("--train_list", default="coco14/train14.txt", type=str)
parser.add_argument("--val_list", default="coco14/val14.txt", type=str)
parser.add_argument("--infer_list", default="coco14/train14.txt", type=str,
                    help="voc12/train_aug.txt to train a fully supervised model, "
                         "voc12/train.txt or voc12/val.txt to quickly check the quality of the labels.")
parser.add_argument("--chainer_eval_set", default="train", type=str)

# Class Activation Map
parser.add_argument("--cam_network", default="net.resnet50_cam", type=str)
parser.add_argument("--cam_crop_size", default=512, type=int)
parser.add_argument("--cam_batch_size", default=2, type=int) # original: 16
parser.add_argument("--cam_num_epoches", default=5, type=int)
parser.add_argument("--cam_learning_rate", default=0.1, type=float)
parser.add_argument("--cam_weight_decay", default=1e-4, type=float)
parser.add_argument("--cam_eval_thres", default=0.15, type=float)
parser.add_argument("--cam_scales", default=(1.0, 0.5, 1.5, 2.0),
                    help="Multi-scale inferences")

parser.add_argument("--cam_weights_name", default="sess/res50_cam_coco.pth", type=str)

parser.add_argument("--cam_out_dir", default="result/cam_RIB_coco", type=str)

parser.add_argument("--RIB_iter", default=10, type=int)
parser.add_argument("--RIB_lr", default=0.000005, type=float)
parser.add_argument("--RIB_batch", default=15, type=int)
parser.add_argument("--score_th", default=0.5, type=float)
parser.add_argument("--stop_th", default=1000, type=int)
parser.add_argument("--p_disjoint", default=1, type=float)
parser.add_argument("--pooling", default='gndp', type=str, help='gap, gndp')

parser.add_argument("--explode_th", default=0.3, type=float)
parser.add_argument("--explode_ratio", default=0.8, type=float)

args = parser.parse_args()
torch.set_num_threads(1)
if not os.path.exists(args.cam_out_dir):
    os.makedirs(args.cam_out_dir)



def save_npy(outputs, size, img_name, label):
    strided_size = imutils.get_strided_size(size, 4)
    strided_up_size = imutils.get_strided_up_size(size, 16)

    strided_cam = torch.sum(torch.stack(
        [F.interpolate(torch.unsqueeze(o, 0), strided_size, mode='bilinear', align_corners=False)[0] for o
         in outputs]), 0)

    highres_cam = [F.interpolate(torch.unsqueeze(o, 1), strided_up_size,
                                 mode='bilinear', align_corners=False) for o in outputs]
    highres_cam = torch.sum(torch.stack(highres_cam, 0), 0)[:, 0, :size[0], :size[1]]
    valid_cat = torch.nonzero(label[0])[:, 0].cpu()
    strided_cam = strided_cam[valid_cat]
    strided_cam /= F.adaptive_max_pool2d(strided_cam, (1, 1)) + 1e-5

    highres_cam = highres_cam[valid_cat]
    highres_cam /= F.adaptive_max_pool2d(highres_cam, (1, 1)) + 1e-5

    # save cams
    np.save(os.path.join(args.cam_out_dir, img_name + '.npy'),
            {"keys": valid_cat, "cam": strided_cam.cpu(), "high_res": highres_cam.cpu().numpy()})




def raw_logit_loss_supp_others_norm(logits, labels):
    target_class = logits * labels
    target_loss = target_class[:6].sum()
    loss = - target_class.sum()
    return loss, target_loss




def RIB_learn(img, label, pack_RIB, optimizer, model, outputs_cam, RIB_step, start_target_loss, process_id, pack_target):
    logits_total, labels_total = [], []
    valid_cat = torch.nonzero(label[0])[:, 0].cpu()

    before_cam = []

    for img_idx, img_target in enumerate(img):
        cam, indss = model(img_target[0].cuda(), cats=valid_cat) # img: 2x3xwxh, cam: 2x21xw'xh'

        cam_flip_one = F.relu(cam)
        cam_flip_one = cam_flip_one[0] + cam_flip_one[1].flip(-1)
        before_cam.append(cam_flip_one.detach())


        if RIB_step == 0:

            ##### flip !!!!
            outputs_cam.append(cam_flip_one.data.cpu())
        else:
            outputs_cam[img_idx] += cam_flip_one.data.cpu()

        if RIB_step > 0 and img_idx == 0:
            cam_interest = cam_flip_one.data.cpu()[valid_cat]
            cam_norm = (cam_interest / (F.adaptive_max_pool2d(cam_interest, (1, 1)) + 1e-5)) > args.explode_th
            cam_norm = torch.clip(cam_norm.sum(axis=0), 0, 1)
            if cam_norm.sum() / cam_norm.shape[0] / cam_norm.shape[1] > args.explode_ratio:
                return outputs_cam, True, start_target_loss

        if img_idx != 1:
            if args.pooling == 'gap':
                logits_total.append(torchutils.gap2d(cam, keepdims=True)[:, :, 0, 0])
            elif args.pooling == 'gndp':
                logits_total.append(torchutils.gndp2d(cam, keepdims=True, valid_cat=valid_cat, score_th=args.score_th)[:, :, 0, 0])


        if img_idx != 1:
            labels_total.append(label)
            labels_total.append(label)
    if RIB_step == 0:
        model.init_cam = before_cam

    img_RIB = pack_RIB['img']
    label_RIB = pack_RIB['label'].cuda(non_blocking=True)
    cam_RIB = model(img_RIB.cuda())
    if args.pooling == 'gap':
        logits_total.append(torchutils.gap2d(cam_RIB, keepdims=True)[:, :, 0, 0])
    elif args.pooling == 'gndp':
        logits_total.append(torchutils.gndp2d(cam_RIB, keepdims=True, valid_cat=pack_RIB['label'], score_th=args.score_th, is_RIB=True)[:, :, 0, 0])

    labels_total.append(label_RIB)

    logits_total = torch.cat(logits_total, dim=0)
    labels_total = torch.cat(labels_total, dim=0)
    loss, target_loss = raw_logit_loss_supp_others_norm(logits_total, labels_total)
    if RIB_step == 0:
        start_target_loss = torch.abs(target_loss).item()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if torch.abs(target_loss) > args.stop_th:
        stop_sign=True
    else:
        stop_sign=False
    return outputs_cam, stop_sign, start_target_loss

def _work(process_id, model, dataset, model_state_dict, args):
    databin = dataset[process_id]
    n_gpus = torch.cuda.device_count()
    data_loader = DataLoader(databin, shuffle=False, num_workers=0, pin_memory=True)
    print("dcpu", args.num_workers // n_gpus)
    count = 0
    print(n_gpus)
    with cuda.device(process_id%n_gpus):
        model.cuda()
        for iter, pack in enumerate(data_loader):
            count += 1

            img = pack['img']   # scale: 1, 0.5, 1.5, 2.0
            label = pack['label'].cuda(non_blocking=True)  # 1, 20
            size = pack['size']
            img_name = pack['name'][0]
            n_classes = len(list(torch.nonzero(pack['label'][0])[:, 0]))
            if n_classes == 0:
                print("no class", img_name)
                valid_cat = torch.nonzero(pack['label'][0])[:, 0]
                np.save(os.path.join(args.cam_out_dir, img_name + '.npy'),
                        {"keys": valid_cat})

                continue

            model.train()
            model.load_state_dict(model_state_dict, strict=True)

            param_groups = model.trainable_parameters()
            optimizer = torchutils.PolyOptimizer([
                {'params': param_groups[0], 'lr': args.RIB_lr, 'weight_decay': args.cam_weight_decay},
                {'params': param_groups[1], 'lr': 10 * args.RIB_lr, 'weight_decay': args.cam_weight_decay},
            ], lr=args.RIB_lr, weight_decay=args.cam_weight_decay, max_step=args.RIB_iter)

            train_dataset = coco14.dataloader.COCO14RIBDataset(args.train_list,
                                             voc12_root=args.voc12_root,
                                             resize_long=(320, 640), hor_flip=True,
                                             crop_size=512, crop_method="random", image_id=img_name, disjoint_prob=args.p_disjoint)
            train_data_loader = DataLoader(train_dataset, batch_size=args.RIB_batch,
                                           shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
            outputs_cam = []
            model.init_cam = []
            start_target_loss = 0
            for RIB_step, pack_RIB in enumerate(train_data_loader):
                if RIB_step == args.RIB_iter:
                    break

                outputs_cam, stop, start_target_loss = RIB_learn(img, label, pack_RIB, optimizer, model, outputs_cam, RIB_step, start_target_loss, process_id, pack)
                if stop:
                    break
            save_npy(outputs=outputs_cam, size=size, img_name=img_name, label=pack['label'])






if __name__ == '__main__':
    model = getattr(importlib.import_module(args.cam_network), 'CAM')(coco=True)
    model_state_dict = torch.load(args.cam_weights_name + '.pth')


    n_gpus = torch.cuda.device_count() * 3
    print(n_gpus)
    dataset = coco14.dataloader.COCO14ClassificationDatasetMSF(args.infer_list,
                                                               coco14_root=args.voc12_root, scales=args.cam_scales)
    dataset = torchutils.split_dataset(dataset, n_gpus)
    multiprocessing.spawn(_work, nprocs=n_gpus, args=(model, dataset, model_state_dict, args), join=True)