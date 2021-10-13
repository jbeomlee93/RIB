
import numpy as np
import os
from chainercv.datasets import VOCSemanticSegmentationDataset
from chainercv.evaluations import calc_semantic_segmentation_confusion
from PIL import Image


def run(args):
    # labels = [dataset.get_example_by_keys(i, (1,))[0] for i in range(len(dataset))]
    ids = open('coco14/train14.txt').readlines()
    ids = [i.split('\n')[0] for i in ids]
    preds = []
    labels = []
    n_images = 0

    for i, id in enumerate(ids):
        label = np.array(Image.open('Dataset/coco_2014/coco_seg_anno/%s.png' % id))
        n_images += 1
        # print(os.path.join(args.cam_out_dir, id + '.npy'))
        cam_dict = np.load(os.path.join(args.cam_out_dir, id + '.npy'), allow_pickle=True).item()
        if not ('high_res' in cam_dict):
            preds.append(np.zeros_like(label))
            labels.append(label)
            continue
        cams = cam_dict['high_res']
        cams = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.cam_eval_thres)
        keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')
        cls_labels = np.argmax(cams, axis=0)
        cls_labels = keys[cls_labels]
        preds.append(cls_labels.copy())
        labels.append(label)
        xx, yy = cls_labels.shape, label.shape
        if xx[0] != yy[0]:
            print(id, xx, yy)

    confusion = calc_semantic_segmentation_confusion(preds, labels)
    np.save('confusion_cam_crf.npy', np.array(confusion))
    print(confusion, confusion.shape)
    gtj = confusion.sum(axis=1)
    resj = confusion.sum(axis=0)
    gtjresj = np.diag(confusion)
    denominator = gtj + resj - gtjresj
    iou = gtjresj / denominator
    print(iou)
    # print(gtj)
    # print(resj)
    # print(confusion)
    # print("Eval with only %s images" % n_images)


    print("threshold:", args.cam_eval_thres, 'miou:', np.nanmean(iou), "i_imgs", n_images)
    # print({'iou': iou, 'miou': np.nanmean(iou)})
    # print(resj.sum(), resj[1:].sum())
    # print('among_predfg_bg', float((gtj[1:].sum()-confusion[1:,1:].sum())/(gtj[1:].sum())))
    print('among_predfg_bg', float((resj[1:].sum()-confusion[1:,1:].sum())/(resj[1:].sum())))

    # print('among_predfg_bg', float((confusion[1:].sum()-confusion[1:][1:].sum())/(confusion[1:].sum())))

    return np.nanmean(iou)