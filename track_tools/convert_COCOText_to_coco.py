"""
https://github.com/xingyizhou/CenterTrack
Modified by Weijia Wu
"""

import math
import os
import numpy as np
import json
from PIL import Image
from track_tools.utils import remove_all,split
import cv2
DATA_PATH = '/share/wuweijia/Data/COCOTextV2/'
OUT_PATH = DATA_PATH + 'annotations_coco_rotate/'
SPLITS = [ 'train']
DEBUG = False

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

def load_func(fpath):
    print('fpath', fpath)
    assert os.path.exists(fpath)
    with open(fpath,'r') as fid:
        lines = fid.readlines()
    records =[json.loads(line.strip('\n')) for line in lines]
    return records
def find_min_rect_angle(vertices):
    '''find the best angle to rotate poly and obtain min rectangle
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
    Output:
        the best angle <radian measure>
    '''
#     print(vertices)
#     try:
#         x1, y1, x2, y2, x3, y3, x4, y4 = vertices
#     except:
#         print(vertices)
#         assert False
    
#     lin = []
#     point = []
#     for i in range(4):
#         lin.append(cal_distance(vertices[i*2], vertices[i*2+1], vertices[(i*2+2)%8], vertices[(i*2+3)%8]))
#         point.append([vertices[i*2], vertices[i*2+1], vertices[(i*2+2)%8], vertices[(i*2+3)%8]])
     
#     idx = lin.index(max(lin))
#     a1,b1,a2,b2 = point[idx]
#     if (a2-a1) == 0:
#         angle_interval = 1
#         angle_list = list(range(0, 90, angle_interval))
#     else:
#         tan = (b2-b1)/(a2-a1)
#         if tan < 0:
#             angle_interval = 1
#             angle_list = list(range(0, 90, angle_interval))
#         else:
#             angle_interval = 1
#             angle_list = list(range(-90, 0, angle_interval))
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
#     print(v)
#     print(anchor)
    if anchor is None:
#         anchor = v[:, :1]
        anchor = np.array([[v[0].sum()],[v[1].sum()]])/4
    rotate_mat = get_rotate_mat(theta)
    res = np.dot(rotate_mat, v - anchor)
    return (res + anchor).T.reshape(-1)

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

def get_rotate_mat(theta):
    '''positive theta value means rotate clockwise'''
    return np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])


def cal_distance(x1, y1, x2, y2):
    '''calculate the Euclidean distance'''
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def get_rotate(box):
    # box : x1,y2...,x3,y3
    theta = find_min_rect_angle(box)
    
#     anchor = np.array([box[::2].sum()/4,box[1::2].sum()/4])
    
    rotate_mat = get_rotate_mat(theta)
    rotated_vertices = rotate_vertices(box, theta)
    x_min, x_max, y_min, y_max = get_boundary(rotated_vertices)
    return np.array([x_min, y_min,x_max-x_min , y_max-y_min]),theta

if __name__ == '__main__':
    if not os.path.exists(OUT_PATH):
        os.mkdir(OUT_PATH)
    for split in SPLITS:
        ann_path = DATA_PATH + "cocotext.json"
        with open(ann_path,'r',encoding='utf-8-sig') as load_f:
            gt = json.load(load_f)
        annns = {}
        for a in gt["anns"]:
            one_anns = gt["anns"][a]
            if one_anns["image_id"] not in annns:
                annns[one_anns["image_id"]] = [one_anns]
            else:
                annns[one_anns["image_id"]].append(one_anns)

#         image_list = os.listdir(data_path)
#         print(len(image_list))
        
        out_path = OUT_PATH + '{}.json'.format(split)
        out = {'images': [], 'annotations': [], 'categories': [{'id': 1, 'name': 'person'}]}
        
        image_cnt = 0
        ann_cnt = 0
        video_cnt = 0
        for idx,a in enumerate(gt["imgs"]):
            try:
                boxs = annns[int(a)]
            except:
                continue
                
            one_imgs = gt["imgs"][a]
#             image = cv2.imread("./train2014/"+ one_imgs["file_name"])
    
            image_cnt += 1
            image_info = {'file_name': one_imgs["file_name"], 
                          'id': image_cnt,
                          'height': one_imgs["height"], 
                          'width': one_imgs["width"]}
            out['images'].append(image_info)
            

            if split != 'test':
                for box in boxs:
                    box_ = box["bbox"]
                    ann_cnt += 1
                    # [526.0, 169.1, 524.9, 151.6, 520.6, 150.4, 521.8, 169.8]
#                     print(box["mask"])
#                     print(box.keys())
#                     assert False
                    points = np.array([int(i) for i in np.array(box["mask"])])
                    try:
                        points = cv2.minAreaRect(points.reshape((int(len(points)/2), 2)))
                    except:
                        print(points)
                        print(int(len(points)/2))
                        assert False
                    # 获取矩形四个顶点，浮点型
                    points = cv2.boxPoints(points).reshape((-1))
        
                    box, rotate = get_rotate(points)
                    
                    ann = {'id': ann_cnt,
                         'category_id': 1,
                         'image_id': image_cnt,
                         'rotate': float(rotate),
                         'bbox_vis': box.tolist(),
                         'bbox': box.tolist(),
                         'area': box[2]*box[3],
                         'iscrowd': 0}

                    out['annotations'].append(ann)
        print('loaded {} for {} images and {} samples'.format(split, len(out['images']), len(out['annotations'])))
        json.dump(out, open(out_path, 'w'))
