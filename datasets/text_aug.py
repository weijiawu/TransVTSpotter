# Modified by Weijia
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import numpy as np
from pathlib import Path
from torch.utils import data
import torch
import torch.utils.data
from pycocotools import mask as coco_mask
from PIL import Image
import math
import cv2
try:
    import xml.etree.cElementTree as ET  # 解析xml的c语言版的模块
except ImportError:
    import xml.etree.ElementTree as ET
from datasets.torchvision_datasets import CocoDetection as TvCocoDetection
from util.misc import get_local_rank, get_local_size
import datasets.transforms as T
import os

def get_rotate_mat(theta):
    '''positive theta value means rotate clockwise'''
    return np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])

def rotate_vertices(vertices, theta, anchor=None):
    '''rotate vertices around anchor
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
        theta   : angle in radian measure
        anchor  : fixed position during rotation
    Output:
        rotated vertices <numpy.ndarray, (8,)>
    '''
    v = vertices.reshape((4, 2)).T
    if anchor is None:
#         anchor = v[:, :1]
        anchor = np.array([[v[0].sum()],[v[1].sum()]])/4
    rotate_mat = get_rotate_mat(theta)
    res = np.dot(rotate_mat, v - anchor)
    return (res + anchor).T.reshape(-1)

def rotate_img(img, ann, angle_range=15):
    '''rotate image [-10, 10] degree to aug data
    Input:
        img         : PIL Image
        vertices    : vertices of text regions <numpy.ndarray, (n,8)>
        angle_range : rotate range
    Output:
        img         : rotated PIL Image
        new_vertices: rotated vertices
    '''
    center_x = (img.width - 1) / 2
    center_y = (img.height - 1) / 2
    angle = angle_range * (np.random.rand() * 2 - 1)
    img = img.rotate(angle, Image.BILINEAR)
    vertices = ann["bbox"]
    new_vertices = np.zeros(vertices.shape)
    for i, vertice in enumerate(vertices):
        new_vertices[i, :] = rotate_vertices(vertice, -angle / 180 * math.pi, np.array([[center_x], [center_y]]))
    ann["bbox"] = new_vertices
    return img, ann

def getBboxesAndLabels_icd13(height, width, annotations):
    bboxes = []
    labels = []
    polys = []
    IDs = []
    rotates = []
    # points_lists = [] # does not contain the ignored polygons.
    for annotation in annotations:
        object_boxes = []
        for point in annotation:
            object_boxes.append([int(point.attrib["x"]), int(point.attrib["y"])])

        points = np.array(object_boxes).reshape((-1))
        points = cv2.minAreaRect(points.reshape((4, 2)))
        # 获取矩形四个顶点，浮点型
        points = cv2.boxPoints(points).reshape((-1))
#         box, rotate = get_rotate(points)
        
#         x, y, w, h = cv2.boundingRect(points)
#         box = np.array([x, y, x + w, y + h])
#         label = annotation.attrib["Transcription"]

#         Transcription = annotation.attrib["Transcription"]
#         if "?" in Transcription or "#" in Transcription or "55" in Transcription:
#             continue
        
        quality = annotation.attrib["Quality"]
        Transcription = annotation.attrib["Transcription"]
        if "?" in Transcription:
            continue
#         if quality == "LOW":
#             continue

        bboxes.append(points)
        IDs.append(annotation.attrib["ID"])


    return np.array(bboxes), IDs

def adjust_box_sort(box):
    start = -1
    _box = list(np.array(box).reshape(-1,2))
    min_x = min(box[0::2])
    min_y = min(box[1::2])
    _box.sort(key=lambda x:(x[0]-min_x)**2+(x[1]-min_y)**2)
    start_point = list(_box[0])
    for i in range(0,8,2):
        x,y = box[i],box[i+1]
        if [x,y] == start_point:
            start = i//2
            break

    new_box = []
    new_box.extend(box[start*2:])
    new_box.extend(box[:start*2])
    return np.array(new_box)


def parse_xml(annotation_path,video_path):
    utf8_parser = ET.XMLParser(encoding='gbk')
    with open(annotation_path, 'r', encoding='gbk') as load_f:
        tree = ET.parse(load_f, parser=utf8_parser)
    root = tree.getroot()  # 获取树型结构的根
    
    bboxess, IDss, rotatess = [], [] , []
    for idx,child in enumerate(root):
        image_path = os.path.join(video_path, child.attrib["ID"] + ".jpg")
        try:
            img = cv2.imread(image_path)
            height, width = img.shape[:2]
        except:
            print(image_path+"is None")
#             continue
            assert False
        bboxes, IDs = \
            getBboxesAndLabels_icd13(height, width, child)
        bboxess.append(bboxes) 
        IDss.append(IDs)
    return np.array(bboxess), np.array(IDss)

def find_min_rect_angle(vertices):
    '''find the best angle to rotate poly and obtain min rectangle
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
    Output:
        the best angle <radian measure>
    '''
            
    angle_interval = 1
    angle_list = list(range(-90, 90, angle_interval))
    vertices = adjust_box_sort(vertices)
    area_list = []
    for theta in angle_list:
        rotated = rotate_vertices(vertices, theta / 180 * math.pi)
        x1, y1, x2, y2, x3, y3, x4, y4 = rotated
        temp_area = (max(x1, x2, x3, x4) - min(x1, x2, x3, x4)) * \
                    (max(y1, y2, y3, y4) - min(y1, y2, y3, y4))
        area_list.append(temp_area)

    sorted_area_index = sorted(list(range(len(area_list))), key=lambda k: area_list[k])
    min_error = float('inf')
    best_index = -1
    rank_num = 10
    # find the best angle with correct orientation
    for index in sorted_area_index[:rank_num]:
        rotated = rotate_vertices(vertices, angle_list[index] / 180 * math.pi)
        temp_error = cal_error(rotated)
        if temp_error < min_error:
            min_error = temp_error
            best_index = index
    return angle_list[best_index] / 180 * math.pi

def get_rotate(box):
    # box : x1,y2...,x3,y3
    theta = find_min_rect_angle(box)
    rotate_mat = get_rotate_mat(theta)
    rotated_vertices = rotate_vertices(box, theta)
    x_min, x_max, y_min, y_max = get_boundary(rotated_vertices)
    return np.array([x_min, y_min,x_max-x_min , y_max-y_min]),theta

def cal_error(vertices):
    '''default orientation is x1y1 : left-top, x2y2 : right-top, x3y3 : right-bot, x4y4 : left-bot
    calculate the difference between the vertices orientation and default orientation
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
    Output:
        err     : difference measure
    '''
    x_min, x_max, y_min, y_max = get_boundary(vertices)
    x1, y1, x2, y2, x3, y3, x4, y4 = vertices
    err = cal_distance(x1, y1, x_min, y_min) + cal_distance(x2, y2, x_max, y_min) + \
          cal_distance(x3, y3, x_max, y_max) + cal_distance(x4, y4, x_min, y_max)
    return err

def get_boundary(vertices):
    '''get the tight boundary around given vertices
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
    Output:
        the boundary
    '''
    x1, y1, x2, y2, x3, y3, x4, y4 = vertices
    x_min = min(x1, x2, x3, x4)
    x_max = max(x1, x2, x3, x4)
    y_min = min(y1, y2, y3, y4)
    y_max = max(y1, y2, y3, y4)
    return x_min, x_max, y_min, y_max

def cal_distance(x1, y1, x2, y2):
    '''calculate the Euclidean distance'''
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def cnvert_to_rotate(ann):
    box,ID = ann["bbox"], ann["ID"]
    
    bbox = []
    rotates = []
    area = []
    for points in box:
        box, rotate = get_rotate(points)
        bbox.append(box)
        rotates.append(rotate)
        area.append((box[2])*(box[3]))
    return {"rotate":rotates,"bbox":bbox,"ID":ID,"area":area}

def crop_img(img, ann, length):
    '''crop img patches to obtain batch and augment
    Input:
        img         : PIL Image
        vertices    : vertices of text regions <numpy.ndarray, (n,8)>
        labels      : 1->valid, 0->ignore, <numpy.ndarray, (n,)>
        length      : length of cropped image region
    Output:
        region      : cropped image region
        new_vertices: new vertices in cropped region
    '''
    vertices = ann["bbox"]
    h, w = img.height, img.width
    # confirm the shortest side of image >= length
    if h >= w and w < length:
        img = img.resize((length, int(h * length / w)), Image.BILINEAR)
    elif h < w and h < length:
        img = img.resize((int(w * length / h), length), Image.BILINEAR)
    ratio_w = img.width / w
    ratio_h = img.height / h
    assert (ratio_w >= 1 and ratio_h >= 1)

    new_vertices = np.zeros(vertices.shape)
    if vertices.size > 0:
        new_vertices[:, [0, 2, 4, 6]] = vertices[:, [0, 2, 4, 6]] * ratio_w
        new_vertices[:, [1, 3, 5, 7]] = vertices[:, [1, 3, 5, 7]] * ratio_h

    # find random position
    remain_h = img.height - length
    remain_w = img.width - length
    flag = True
    cnt = 0
    while flag and cnt < 1000:
        cnt += 1
        start_w = int(np.random.rand() * remain_w)
        start_h = int(np.random.rand() * remain_h)
        flag = is_cross_text([start_w, start_h], length, new_vertices[:, :])
    box = (start_w, start_h, start_w + length, start_h + length)
    region = img.crop(box)
    if new_vertices.size == 0:
        return region, new_vertices

    new_vertices[:, [0, 2, 4, 6]] -= start_w
    new_vertices[:, [1, 3, 5, 7]] -= start_h
    ann["bbox"] = new_vertices
    return region, ann

def adjust_height(img, ann, ratio=0.2):
    '''adjust height of image to aug data
    Input:
        img         : PIL Image
        vertices    : vertices of text regions <numpy.ndarray, (n,8)>
        ratio       : height changes in [0.8, 1.2]
    Output:
        img         : adjusted PIL Image
        new_vertices: adjusted vertices
    '''
    vertices = ann["bbox"]
    ratio_h = 1 + ratio * (np.random.rand() * 2 - 1)
    old_h = img.height
    new_h = int(np.around(old_h * ratio_h))
    img = img.resize((img.width, new_h), Image.BILINEAR)

    new_vertices = vertices.copy()
    if vertices.size > 0:
        new_vertices[:, [1, 3, 5, 7]] = vertices[:, [1, 3, 5, 7]] * (new_h / old_h)
        
    ann["bbox"] = new_vertices
    return img, ann

class ICDAR15_Video(data.Dataset):
    def __init__(self, DATA_PATH, split, transforms, return_masks, cache_mode=False, local_rank=0, local_size=1):
        super(ICDAR15_Video, self).__init__()
        
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)
        
        if split == "test":
            data_path = os.path.join(DATA_PATH, 'test/frames')
            ann_path_ = os.path.join(DATA_PATH, 'test/gt')
        else:
            data_path = os.path.join(DATA_PATH, 'train/frames')
            ann_path_ = os.path.join(DATA_PATH, 'train/gt')
        
        seqs = os.listdir(data_path)
        self.image_list = []
        self.ann_list = []
        self.frame_list = []
        image_cnt = 0
        ann_cnt = 0
        video_cnt = 0
        
        for seq in sorted(seqs):
            if '.DS_Store' in seq:
                continue
            video_cnt += 1  # video sequence number.
            seq_path = os.path.join(data_path, seq)
            ann_path = os.path.join(ann_path_, seq + "_GT.xml")
            images = os.listdir(seq_path)
            num_images = len([image for image in images if 'jpg' in image])  # half and half
            
            if split != 'test':
                bboxess, IDss = parse_xml(ann_path,os.path.join(data_path,seq))
                    
            for i in range(num_images):
                image_path = os.path.join(data_path, '{}/{}.jpg'.format(seq, i + 1))
                self.image_list.append(image_path)
                frame_id = i + 1
                self.frame_list.append(frame_id)
                if split != 'test':
                    bboxes,IDs = bboxess[i],IDss[i]    
                    self.ann_list.append({"bbox":bboxes,"ID":IDs})
                                          
    
    def __len__(self):
        return len(self.image_list)
    
    def get_image(self, path):
        return Image.open(path).convert('RGB')
    
    def __getitem__(self, idx):
        path = self.image_list[idx]
        ann = self.ann_list[idx]
                                          
        img = self.get_image(path)
        
        img, ann = adjust_height(img, ann)
        img, ann = rotate_img(img, ann)
#         img, ann = crop_img(img, ann, self.length)
        
        ann = cnvert_to_rotate(ann)
                                         
        target = {'image_id': idx, 'frame_id': self.frame_list[idx], 'annotations': ann}
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

        boxes = [obj for obj in anno["bbox"]]
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]

        classes = [1 for obj in anno["bbox"]]
        classes = torch.tensor(classes, dtype=torch.int64)
        
        rotates = [obj for obj in anno["rotate"]]
        rotates = torch.tensor(rotates, dtype=torch.float)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        rotates = rotates[keep]

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

        area = torch.tensor([obj for obj in anno["area"]])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno["bbox"]])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target

    
    
    
def make_mot_transforms(image_set, args):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train' and not args.eval:
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomResize(scales, max_size=1333),
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
        "train": (root / "train/frames", root / "annotations_coco_rotate" / 'train.json'),
        "val": (root / "test/frames", root / "annotations_coco_rotate" / 'test.json'),
        "test": (root / "test/frames", root / "annotations_coco_rotate" / 'test.json')
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = ICDAR15_Video(root, image_set, transforms=make_mot_transforms(image_set, args), return_masks=args.masks,
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
    import numpy as np
    import math
    import cv2
    from util import box_ops
    from torchvision import transforms
    root = '/share/wuweijia/Data/ICDAR2015_video'
    img_folder = root + "train/frames"
    ann_file = root +  'annotations_coco_rotate/train.json'
    image_set = "train"
    
    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    parser.add_argument('--eval', action='store_true')
    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')
    args = parser.parse_args()
    
    dataset = ICDAR15_Video(root, split="train", transforms=make_mot_transforms(image_set, args), return_masks=args.masks,
                            cache_mode=args.cache_mode, local_rank=get_local_rank(), local_size=get_local_size())
    
    train_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=0)
    
    for i, (img, target) in enumerate(train_loader):
#         print()
        if i < 1000:
            continue
        if i > 2000:
            break
        img = img[0]
        h, w = img.shape[1:]
        
        img = ((img.to(torch.float)).numpy().transpose(1, 2, 0) * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406])*255
#         cv2.imwrite('./output/show/img{}.png'.format(i), img)
        boxes = target["boxes"]
        labels = target["labels"][0]
        rotate = target["rotate"][0]
        if len(labels)!=len(rotate):
            print(labels)
            print(rotate)
            assert False
        boxes = boxes * torch.tensor([w, h, w, h], dtype=torch.float32)
        boxes = box_ops.box_cxcywh_to_xyxy(boxes)
        
        
                
#         w,h = target["size"][0]
#         boxes = boxes * torch.tensor([w, h, w, h], dtype=torch.float32)[0]

        img1 = img.copy() 
#         print(rotate)
        for idx,box in enumerate(boxes[0]):
            try:
                box11=box[0]
            except:
                continue
                
            x_min,y_min, x_max, y_max = int(box[0]),int(box[1]),int(box[2]),int(box[3])
            cv2.rectangle(img1, (int(box[0]),int(box[1])), (int(box[2]),int(box[3])), (0,255,0), 2)
            
            print(rotate[idx])
            rotate_mat = get_rotate_mat(-rotate[idx])
            temp_x = np.array([[x_min, x_max, x_max, x_min]]) - (x_min+x_max)/2
            temp_y = np.array([[y_min, y_min, y_max, y_max]]) - (y_min+y_max)/2
            coordidates = np.concatenate((temp_x, temp_y), axis=0)
            res = np.dot(rotate_mat, coordidates)
            res[0,:] += (x_min+x_max)/2
            res[1,:] += (y_min+y_max)/2
#             detetcion_box = [res[0,0], res[1,0], res[0,1], res[1,1], res[0,2], res[1,2],res[0,3], res[1,3]]
            
            bbox = np.array([int(res[0,0]), int(res[1,0]), int(res[0,1]), int(res[1,1]), int(res[0,2]), int(res[1,2]),int(res[0,3]), int(res[1,3])])
#             print(bbox)
            cv2.drawContours(img1, [bbox.reshape(int(bbox.shape[0] / 2), 2)], -1, (0, 0, 255), 3)
#             print((int(box[0]),int(box[1])))
#             cv2.rectangle(img1, (int(box[0]),int(box[1])), (int(box[2]),int(box[3])), (0,255,0), 4)
            
        cv2.imwrite('./output/show/img{}_vis.png'.format(i), img1)
#         print(w,h)
#         print(target)
#         print(img.shape)
#         print(boxes)
