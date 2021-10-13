# Reducing Information Bottleneck for Weakly Supervised Semantic Segmentation (NeurIPS 2021)

The implementation of Reducing Infromation Bottleneck for Weakly Supervised Semantic Segmentation, Jungbeom Lee, Jooyoung Choi, Jisoo Mok, and Sungroh Yoon, NeurIPS 2021. [[paper](https://arxiv.org/abs/2103.08896)]

## Installation

- We kindly refer to the offical implementation of [IRN](https://github.com/jiwoon-ahn/irn).

## Usage

### Step 1. Prepare Dataset

- Download Pascal VOC dataset [here](https://drive.google.com/file/d/1jhtdjj3xrEp60zO3B7jZ14yxxZkCJMeM/view?usp=sharing).
- Download MS COCO images from the official COCO website [here](https://cocodataset.org/#download).
- Download semantic segmentation annotations for the MS COCO dataset [here](https://drive.google.com/file/d/1pRE9SEYkZKVg0Rgz2pi9tg48j7GlinPV/view?usp=sharing).

- Directory hierarchy 
```
    Dataset
    ├── VOC2012_SEG_AUG       # unzip VOC2012_SEG_AUG.zip           
    ├── coco_2017             # mkdir coco_2017
    │   ├── coco_seg_anno     # included in coco_annotations_semantic.zip
    └── └── JPEGImages        # include train and val images downloaded from the official COCO website
```

### Step 2. Prepare pre-trained classifier
- Pre-trained model used in this paper: [Pascal VOC](https://drive.google.com/file/d/1evRxPD4PhcdGySGFXDoOAtOiUVg4oNXk/view?usp=sharing), [MS COCO](https://drive.google.com/file/d/1SDKjPzzuXR4PX_H1l3Kw5NY8d20FmG4X/view?usp=sharing).
- You can also train your own classifiers following [IRN](https://github.com/jiwoon-ahn/irn).

### Step 3. Generate and evaluate the pseudo ground-truth masks for PASCAL VOC and MS COCO
- PASCAL VOC

```
bash get_pseudo_gt_VOC.sh
```

- MS COCO

```
bash get_pseudo_gt_COCO.sh
```

### Step 4. Train a semantic segmentation network
- To train DeepLab-v2, we refer to [deeplab-pytorch](https://github.com/kazuto1011/deeplab-pytorch). However, this repo contains only COCO pre-trained model. We provide [ImageNet pre-trained model](https://drive.google.com/file/d/14soMKDnIZ_crXQTlol9sNHVPozcQQpMn/view?usp=sharing) for a fair comparison with the other methods.


## Acknowledgment
This code is heavily borrowed from [IRN](https://github.com/jiwoon-ahn/irn), thanks [jiwoon-ahn](https://github.com/jiwoon-ahn)!



