import torch
from torch import multiprocessing, cuda
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.backends import cudnn

import numpy as np
import importlib
import os
import imageio

import coco14.dataloader
from misc import torchutils, indexing
from PIL import Image

cudnn.enabled = True
palette = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.5), (0.0, 0.0, 1.0), (0.0, 0.25, 0.0), (0.0, 0.25, 0.5), (0.0, 0.25, 1.0),
           (0.0, 0.5, 0.0), (0.0, 0.5, 0.25), (0.0, 0.5, 0.5), (0.0, 0.75, 0.0), (0.0, 0.75, 0.25), (0.0, 0.75, 0.5),
           (0.0, 0.75, 0.75), (0.0, 0.75, 1.0), (0.0, 1.0, 0.25), (0.25, 0.0, 0.0), (0.25, 0.0, 0.25), (0.25, 0.0, 0.5),
           (0.25, 0.0, 1.0), (0.25, 0.25, 0.0), (0.25, 0.25, 0.5), (0.25, 0.25, 1.0), (0.25, 0.5, 0.0), (0.25, 0.5, 0.25),
           (0.25, 0.5, 1.0), (0.25, 0.75, 0.0), (0.25, 0.75, 0.25), (0.25, 0.75, 0.5), (0.25, 1.0, 0.0), (0.25, 1.0, 0.75),
           (0.5, 0.0, 0.0), (0.5, 0.0, 0.25), (0.5, 0.0, 0.5), (0.5, 0.0, 0.75), (0.5, 0.25, 0.0), (0.5, 0.25, 1.0),
           (0.5, 0.5, 0.0), (0.5, 0.5, 0.25), (0.5, 0.5, 0.5), (0.5, 0.5, 0.75), (0.5, 0.75, 0.0), (0.5, 0.75, 0.5),
           (0.5, 0.75, 0.75), (0.5, 1.0, 0.0), (0.5, 1.0, 0.25), (0.5, 1.0, 0.5), (0.5, 1.0, 1.0), (0.75, 0.0, 0.0),
           (0.75, 0.0, 0.25), (0.75, 0.0, 1.0), (0.75, 0.25, 0.0), (0.75, 0.25, 1.0), (0.75, 0.5, 0.0), (0.75, 0.5, 0.25),
           (0.75, 0.5, 0.5), (0.75, 0.5, 0.75), (0.75, 0.5, 1.0), (0.75, 0.75, 0.0), (0.75, 0.75, 0.25), (0.75, 0.75, 0.5),
           (0.75, 0.75, 0.75), (0.75, 0.75, 1.0), (0.75, 1.0, 0.0), (0.75, 1.0, 0.25), (0.75, 1.0, 0.5), (0.75, 1.0, 0.75),
           (0.75, 1.0, 1.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.5), (1.0, 0.25, 0.25), (1.0, 0.25, 0.5), (1.0, 0.25, 0.75),
           (1.0, 0.25, 1.0), (1.0, 0.5, 0.0), (1.0, 0.5, 0.25), (1.0, 0.5, 0.5), (1.0, 0.5, 0.75),
           (1.0, 0.5, 1.0), (1.0, 0.75, 0.25), (1.0, 0.75, 0.5), (1.0, 0.75, 0.75)]
palette_new = []

for p in palette:
    for pp in p:
        palette_new.append(int(pp*255))
palette=palette_new
def _work(process_id, model, dataset, args):

    n_gpus = torch.cuda.device_count()
    databin = dataset[process_id]
    data_loader = DataLoader(databin,
                             shuffle=False, num_workers=args.num_workers // n_gpus, pin_memory=False)

    with torch.no_grad(), cuda.device(process_id % n_gpus):

        model.cuda()
        count = 0

        for iter, pack in enumerate(data_loader):
            count += 1
            img_name = coco14.dataloader.decode_int_filename(pack['name'][0])

            orig_img_size = np.asarray(pack['size'])
            edge, dp = model(pack['img'][0].cuda(non_blocking=True), coco=True)

            cam_dict = np.load(args.cam_out_dir + '/' + img_name + '.npy', allow_pickle=True).item()

            if not ('cam' in cam_dict):

                conf = np.zeros_like(pack['img'][0])[0, 0]
                out = Image.fromarray(conf.astype(np.uint8), mode='P')
                out.putpalette(palette)

                out.save(os.path.join(os.path.join(args.sem_seg_out_dir, img_name + '_palette.png')))
                imageio.imsave(os.path.join(args.sem_seg_out_dir, img_name + '.png'), conf.astype(np.uint8))
                continue
            cams = np.power(cam_dict['cam'], args.sem_seg_power)

            keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')

            cam_downsized_values = cams.cuda()
            rw = indexing.propagate_to_edge(cam_downsized_values, edge, beta=args.beta, exp_times=args.exp_times, radius=5)

            rw_up = F.interpolate(rw, scale_factor=4, mode='bilinear', align_corners=False)[..., 0, :orig_img_size[0], :orig_img_size[1]]
            rw_up = rw_up / torch.max(rw_up)


            rw_up_bg = F.pad(rw_up, (0, 0, 0, 0, 1, 0), value=args.sem_seg_bg_thres)
            rw_pred = torch.argmax(rw_up_bg, dim=0).cpu().numpy()

            rw_pred = keys[rw_pred]

            out = Image.fromarray(rw_pred.astype(np.uint8), mode='P')
            out.putpalette(palette)
            out.save(os.path.join(os.path.join(args.sem_seg_out_dir, img_name + '_palette.png')))
            imageio.imsave(os.path.join(args.sem_seg_out_dir, img_name + '.png'), rw_pred.astype(np.uint8))

            if process_id == n_gpus - 1 and iter % (len(databin) // 20) == 0:
                print("%d " % ((5*iter+1)//(len(databin) // 20)), end='')


def run(args):
    model = getattr(importlib.import_module(args.irn_network), 'EdgeDisplacement')()
    print(args.irn_weights_name)
    model.load_state_dict(torch.load(args.irn_weights_name), strict=False)
    model.eval()

    n_gpus = torch.cuda.device_count() * 3

    dataset = coco14.dataloader.COCO14ClassificationDatasetMSF(args.infer_list,
                                                             coco14_root=args.voc12_root,
                                                             scales=(1.0,))
    dataset = torchutils.split_dataset(dataset, n_gpus)

    print("[", end='')
    multiprocessing.spawn(_work, nprocs=n_gpus, args=(model, dataset, args), join=True)
    print("]")

    torch.cuda.empty_cache()
