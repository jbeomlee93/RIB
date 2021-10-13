import argparse
import os
import numpy as np

from misc import pyutils


import torch
torch.set_num_threads(4)


if __name__ == '__main__':

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
    parser.add_argument("--cam_batch_size", default=16, type=int) # original: 16
    parser.add_argument("--cam_num_epoches", default=5, type=int)
    parser.add_argument("--cam_learning_rate", default=0.1, type=float)
    parser.add_argument("--cam_weight_decay", default=1e-4, type=float)
    parser.add_argument("--cam_eval_thres", default=0.15, type=float)
    parser.add_argument("--cam_scales", default=(1.0, 0.5, 1.5, 2.0),
                        help="Multi-scale inferences")

    # Mining Inter-pixel Relations
    parser.add_argument("--conf_fg_thres", default=0.35, type=float) # 0.35
    parser.add_argument("--conf_bg_thres", default=0.17, type=float) # 0.15

    # Inter-pixel Relation Network (IRNet)
    parser.add_argument("--irn_network", default="net.resnet50_irn", type=str)
    parser.add_argument("--irn_crop_size", default=512, type=int)
    parser.add_argument("--irn_batch_size", default=32, type=int)
    parser.add_argument("--irn_num_epoches", default=3, type=int)
    parser.add_argument("--irn_learning_rate", default=0.1, type=float)
    parser.add_argument("--irn_weight_decay", default=1e-4, type=float)

    # Random Walk Params
    parser.add_argument("--beta", default=10)
    parser.add_argument("--exp_times", default=8,
                        help="Hyper-parameter that controls the number of random walk iterations,"
                             "The random walk is performed 2^{exp_times}.")
    parser.add_argument("--ins_seg_bg_thres", default=0.25)
    parser.add_argument("--sem_seg_bg_thres", default=0.22)
    parser.add_argument("--sem_seg_power", default=1.3, type=float)

    # Output Path
    parser.add_argument("--log_name", default="sample_train_eval_coco", type=str)
    parser.add_argument("--cam_weights_name", default="sess/res50_cam_coco.pth", type=str)
    parser.add_argument("--irn_weights_name", default="sess/res50_irn_coco.pth", type=str)

    parser.add_argument("--cam_out_dir", default="result/cam_RIB_coco", type=str)

    parser.add_argument("--ir_label_out_dir", default="result/ir_label_coco",
                        type=str)
    parser.add_argument("--sem_seg_out_dir", default="result/sem_seg_coco", type=str)
    parser.add_argument("--ins_seg_out_dir", default="result/ins_seg_coco", type=str)
    # Step
    parser.add_argument("--train_cam_pass", default=False)
    parser.add_argument("--make_cam_pass", default=False)
    parser.add_argument("--eval_cam_pass", default=True)
    parser.add_argument("--cam_to_ir_label_pass", default=True)
    parser.add_argument("--train_irn_pass", default=True)
    parser.add_argument("--make_ins_seg_pass", default=False)
    parser.add_argument("--eval_ins_seg_pass", default=False)
    parser.add_argument("--make_sem_seg_pass", default=True)  # check power
    parser.add_argument("--eval_sem_seg_pass", default=True)

    args = parser.parse_args()

    os.makedirs("sess", exist_ok=True)
    os.makedirs(args.cam_out_dir, exist_ok=True)
    os.makedirs(args.ir_label_out_dir, exist_ok=True)
    os.makedirs(args.sem_seg_out_dir, exist_ok=True)
    os.makedirs(args.ins_seg_out_dir, exist_ok=True)
    pyutils.Logger(args.log_name + '.log')
    print(vars(args))

    if args.train_cam_pass is True:
        import step.train_cam_coco

        timer = pyutils.Timer('step.train_cam:')
        step.train_cam_coco.run(args)

    if args.make_cam_pass is True:
        import step.make_cam_coco

        timer = pyutils.Timer('step.make_cam:')
        step.make_cam_coco.run(args)

    if args.eval_cam_pass is True:
        import step.eval_cam_coco

        timer = pyutils.Timer('step.eval_cam:')
        final_miou = []
        for i in range(10, 51):
            t = i/100.0
            args.cam_eval_thres = t
            miou = step.eval_cam_coco.run(args)
            final_miou.append(miou)
        print(args.cam_out_dir)
        print(final_miou)
        print(np.max(np.array(final_miou)))
    if args.cam_to_ir_label_pass is True or args.cam_to_ir_label_pass is 'True':
        import step.cam_to_ir_label_coco

        timer = pyutils.Timer('step.cam_to_ir_label:')
        step.cam_to_ir_label_coco.run(args)

    if args.train_irn_pass is True:
        args.num_workers = 0
        import step.train_irn_coco

        timer = pyutils.Timer('step.train_irn:')
        step.train_irn_coco.run(args)

    if args.make_ins_seg_pass is True:
        import step.make_ins_seg_labels

        timer = pyutils.Timer('step.make_ins_seg_labels:')
        step.make_ins_seg_labels.run(args)

    if args.eval_ins_seg_pass is True:
        import step.eval_ins_seg

        timer = pyutils.Timer('step.eval_ins_seg:')
        step.eval_ins_seg.run(args)

    if args.make_sem_seg_pass is True:
        import step.make_sem_seg_labels_coco
        args.sem_seg_bg_thres = float(args.sem_seg_bg_thres)
        timer = pyutils.Timer('step.make_sem_seg_labels:')
        step.make_sem_seg_labels_coco.run(args)

    if args.eval_sem_seg_pass is True:
        import step.eval_sem_seg_coco

        timer = pyutils.Timer('step.eval_sem_seg:')
        step.eval_sem_seg_coco.run(args)

