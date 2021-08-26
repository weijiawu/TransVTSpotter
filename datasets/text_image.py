# Modified by weijia wu
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

from datasets.torchvision_datasets import CocoDetection as TvCocoDetection
from util.misc import get_local_rank, get_local_size
import datasets.transforms as T


class CocoDetection(TvCocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks, cache_mode=False, local_rank=0, local_size=1):
        super(CocoDetection, self).__init__(img_folder, ann_file,
                                            cache_mode=cache_mode, local_rank=local_rank, local_size=local_size)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'frame_id': idx+1,'video_id': 1, 'annotations': target}
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
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)
        
        rotates = [obj["rotate"] for obj in anno]
        rotates = torch.tensor(rotates, dtype=torch.float)

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
        rotates = rotates[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["rotate"] = rotates
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
#                     T.RandomResize([400, 500, 600]),
#                     T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'test':
        return T.Compose([
            T.RandomResize([800], max_size=1920),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided Crowdhuman path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": (root / "train_images", root / "annotations_coco_rotate" / 'train.json'),
        "test": (root / "test_images", root / "annotations_coco_rotate" / 'test.json'),
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set), return_masks=args.masks,
                            cache_mode=args.cache_mode, local_rank=get_local_rank(), local_size=get_local_size())
    return dataset


def get_rotate_mat(theta):
	'''positive theta value means rotate clockwise'''
	return np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])

if __name__ == "__main__":
    import torch
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse
    from util import box_ops
    import cv2
    import math
    import numpy as np
    from torchvision import transforms
#     root = "/share/wuweijia/Data/ICDAR17/"
#     root = "/share/wuweijia/Data/COCOTextV2/"
#     img_folder = root + "train_image"
#     ann_file = root +  'annotations_coco_rotate/train.json'
    
    root = "/share/wuweijia/Data/ICDAR2015/"
    img_folder = root + "train_images"
    ann_file = root +  'annotations_coco_rotate/train.json'
    
    image_set = "train"
    
    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    parser.add_argument('--eval', action='store_true')
    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')
    args = parser.parse_args()
    
    dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set), return_masks=args.masks,
                            cache_mode=args.cache_mode, local_rank=get_local_rank(), local_size=get_local_size())
    
    train_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=0)
    
    for i, (img, target) in enumerate(train_loader):
#         print()
        if i > 500:
            break
        img = img[0]
        h, w = img.shape[1:]
        
        img = ((img.to(torch.float)).numpy().transpose(1, 2, 0) * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406])*255
#         cv2.imwrite('./output/show/img{}.png'.format(i), img)
        boxes = target["boxes"]
        boxes = boxes * torch.tensor([w, h, w, h], dtype=torch.float32)
        boxes = box_ops.box_cxcywh_to_xyxy(boxes)
        rotate = target["rotate"][0]
        
        img1 = img.copy() 
        for idx,box in enumerate(boxes[0]):
            try:
                box11=box[0]
            except:
                continue
#             print(img.shape)
            x_min,y_min, x_max, y_max = int(box[0]),int(box[1]),int(box[2]),int(box[3])
            cv2.rectangle(img1, (int(box[0]),int(box[1])), (int(box[2]),int(box[3])), (0,255,0), 4)
            print(rotate[idx])
            rotate_mat = get_rotate_mat(-rotate[idx])
            temp_x = np.array([[x_min, x_max, x_max, x_min]]) - (x_min+x_max)/2
            temp_y = np.array([[y_min, y_min, y_max, y_max]]) - (y_min+y_max)/2
            coordidates = np.concatenate((temp_x, temp_y), axis=0)
            res = np.dot(rotate_mat, coordidates)
            res[0,:] += (x_min+x_max)/2
            res[1,:] += (y_min+y_max)/2
            
            bbox = np.array([int(res[0,0]), int(res[1,0]), int(res[0,1]), int(res[1,1]), int(res[0,2]), int(res[1,2]),int(res[0,3]), int(res[1,3])])
#             print(bbox)
            cv2.drawContours(img1, [bbox.reshape(int(bbox.shape[0] / 2), 2)], -1, (0, 0, 255), 3)
    
#             print((int(box[0]),int(box[1])))
            
            
        cv2.imwrite('./output/show/img{}_vis.png'.format(i), img1)
#         print(w,h)
#         print(target)
#         print(img.shape)
#         print(boxes)
