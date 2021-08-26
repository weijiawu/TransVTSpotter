# Modified by Weijia
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
"""
COCO dataset which returns image_id for evaluation.
Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path

import torch
import torch.utils.data
from pycocotools import mask as coco_mask

from .torchvision_datasets import CocoDetection as TvCocoDetection
from util.misc import get_local_rank, get_local_size
import datasets.transforms as T


class CocoDetection(TvCocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks, cache_mode=False, local_rank=0, local_size=1):
        super(CocoDetection, self).__init__(img_folder, ann_file,
                                            cache_mode=cache_mode, local_rank=local_rank, local_size=local_size)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)

    def __getitem__(self, idx):
        coco = self.coco
        img_id = self.ids[idx]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']

        img = self.get_image(path)
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        image_id = self.ids[idx]
        assert image_id == img_id
        frame_id = coco.loadImgs(img_id)[0]['frame_id']
        video_id = coco.loadImgs(img_id)[0]['video_id']
        target = {'image_id': image_id, 'frame_id': frame_id,
                  'video_id': video_id, 'annotations': target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        frame_id = target["frame_id"]
        frame_id = torch.tensor([frame_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
#         boxes[:, 0::2].clamp_(min=0, max=w)
#         boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        target["frame_id"] = frame_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
#                     T.RandomSizeCrop(384, 600),
                    T.RandomSizeCrop_MOT(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])
    
    if image_set == 'trainall':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])
    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])
    if image_set == 'test':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])
    raise ValueError(f'unknown {image_set}')

    
    
    
def make_mot_transforms(image_set, args):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train' and not args.eval:
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([800, 1000, 1200]),
#                     T.RandomSizeCrop(384, 600),
                    T.RandomSizeCrop_MOT(800, 1200),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])
    
    if image_set == 'trainall' and not args.eval:
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([800, 1000, 1200]),
#                     T.RandomSizeCrop(384, 600),
                    T.RandomSizeCrop_MOT(800, 1200),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])
    if image_set == 'val' or args.eval:
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])
    if image_set == 'test' or args.eval:
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])
    raise ValueError(f'unknown {image_set}')



    
def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided MOT path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": (root / "images", root / "annotations_coco" / 'train_half.json'),
        "val": (root / "images", root / "annotations_coco" / 'val_half.json'),
        "test": (root / "test", root / "annotations" / 'test.json'),
        "trainall": (root / "train", root / "annotations" / 'train.json'),
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = CocoDetection(img_folder, ann_file, transforms=make_mot_transforms(image_set, args), return_masks=args.masks,
                            cache_mode=args.cache_mode, local_rank=get_local_rank(), local_size=get_local_size())
    return dataset

if __name__ == "__main__":
    import torch
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse
    import cv2
    from torchvision import transforms
    root = "/home/wuweijia/.jupyter/Data/ICDAR2013_video/"
    img_folder = root + "images"
    ann_file = root +  'annotations_coco/train_half.json'
    image_set = "train"
    
    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    parser.add_argument('--eval', action='store_true')
    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')
    args = parser.parse_args()
    
    dataset = CocoDetection(img_folder, ann_file, transforms=make_mot_transforms(image_set, args), return_masks=args.masks,
                            cache_mode=args.cache_mode, local_rank=get_local_rank(), local_size=get_local_size())
    
    train_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=0)
    
    for i, (img, target) in enumerate(train_loader):
#         print()
        if i > 200:
            break
        img = img[0]
        
        img = ((img.to(torch.float)).numpy().transpose(1, 2, 0) * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406])*255
        cv2.imwrite('./output/show/img{}.png'.format(i), img)
        boxes = target["boxes"]
        w,h = target["size"][0]
        boxes = boxes * torch.tensor([w, h, w, h], dtype=torch.float32)[0]

        img1 = img.copy() 
        for box in boxes[0]:
            try:
                box11=box[0]
            except:
                continue
#             print(img.shape)

#             print((int(box[0]),int(box[1])))
            cv2.rectangle(img1, (int(box[0]),int(box[1])), (int(box[0])+int(box[2]),int(box[1])+int(box[3])), (0,255,0), 4)
            
        cv2.imwrite('./output/show/img{}_vis.png'.format(i), img1)
#         print(w,h)
        print(target)
        print(img.shape)
#         print(boxes)