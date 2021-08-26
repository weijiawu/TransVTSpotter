# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.utils.data
from .torchvision_datasets import CocoDetection

from .coco import build as build_coco
from .mot import build as build_mot
from .text import build as build_text
from .text_image import build as build_icdar15
from .text_aug import build as build_text_icdar15
from .mix import build as build_mix
from .Pretrain import build as build_pretrain

def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, CocoDetection):
        return dataset.coco


def build_dataset(image_set, args):
    if args.dataset_file == 'coco':
        return build_coco(image_set, args)
    if args.dataset_file == 'text':
        return build_text(image_set, args)

    if args.dataset_file == 'text_icd15':
        return build_icdar15(image_set, args)
    
    if args.dataset_file == 'mot':
        return build_mot(image_set, args)
    if args.dataset_file == 'mix':
        return build_mix(image_set, args)
    if args.dataset_file == 'pretrain':
        return build_pretrain(image_set, args)
    if args.dataset_file == 'coco_panoptic':
        # to avoid making panopticapi required for coco
        from .coco_panoptic import build as build_coco_panoptic
        return build_coco_panoptic(image_set, args)
    else:
        return build_text(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')

