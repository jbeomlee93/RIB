
import os
import numpy as np
import imageio

from torch import multiprocessing
from torch.utils.data import DataLoader

import coco14.dataloader
from misc import torchutils, imutils
from PIL import Image
import torch

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
def _work(process_id, infer_dataset, args):
    torch.set_num_threads(2)

    visualize_intermediate_cam = False
    databin = infer_dataset[process_id]
    infer_data_loader = DataLoader(databin, shuffle=False, num_workers=0, pin_memory=False)

    for iter, pack in enumerate(infer_data_loader):
        img_name = coco14.dataloader.decode_int_filename(pack['name'][0])
        img = pack['img'][0].numpy()

        try:
            label = imageio.imread(os.path.join(args.ir_label_out_dir, img_name + '.png'))
            print(img_name, "passed")
        except:
            cam_dict = np.load(os.path.join(args.cam_out_dir, img_name + '.npy'), allow_pickle=True).item()

            if not ('high_res' in cam_dict):
                conf = np.zeros_like(img)[:, :, 0]
                out = Image.fromarray(conf.astype(np.uint8), mode='P')
                out.putpalette(palette)
                out.save(os.path.join(os.path.join(args.ir_label_out_dir, img_name + '_palette.png')))
                imageio.imwrite(os.path.join(args.ir_label_out_dir, img_name + '.png'),
                                conf.astype(np.uint8))
                continue
            cams = cam_dict['high_res']
            keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')




            # 1. find confident fg & bg
            fg_conf_cam = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.conf_fg_thres)
            fg_conf_cam = np.argmax(fg_conf_cam, axis=0)


            pred = imutils.crf_inference_label(img, fg_conf_cam, n_labels=keys.shape[0])

            fg_conf = keys[pred]
            bg_conf_cam = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.conf_bg_thres)
            bg_conf_cam = np.argmax(bg_conf_cam, axis=0)
            pred = imutils.crf_inference_label(img, bg_conf_cam, n_labels=keys.shape[0])
            bg_conf = keys[pred]

            # 2. combine confident fg & bg
            conf = fg_conf.copy()
            conf[fg_conf == 0] = 255
            conf[bg_conf + fg_conf == 0] = 0

            out = Image.fromarray(conf.astype(np.uint8), mode='P')
            out.putpalette(palette)
            out.save(os.path.join(os.path.join(args.ir_label_out_dir, img_name + '_palette.png')))
            imageio.imwrite(os.path.join(args.ir_label_out_dir, img_name + '.png'),
                            conf.astype(np.uint8))


            if process_id == args.num_workers - 1 and iter % (len(databin) // 20) == 0:
                print("%d " % ((5 * iter + 1) // (len(databin) // 20)), end='')

def run(args):
    dataset = coco14.dataloader.COCO14ImageDataset(args.train_list, coco14_root=args.voc12_root, img_normal=None, to_torch=False)
    dataset = torchutils.split_dataset(dataset, 64)

    print('[ ', end='')
    multiprocessing.spawn(_work, nprocs=64, args=(dataset, args), join=True)
    print(']')
