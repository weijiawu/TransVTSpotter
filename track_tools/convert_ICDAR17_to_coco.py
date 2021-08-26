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


def load_func(fpath):
    print('fpath', fpath)
    assert os.path.exists(fpath)
    with open(fpath,'r') as fid:
        lines = fid.readlines()
    records =[json.loads(line.strip('\n')) for line in lines]
    return records


def cal_distance(x1, y1, x2, y2):
    '''calculate the Euclidean distance'''
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def find_min_rect_angle(vertices):
    '''find the best angle to rotate poly and obtain min rectangle
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
    Output:
        the best angle <radian measure>
    '''
#     x1, y1, x2, y2, x3, y3, x4, y4 = vertices
    
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

def get_rotate_mat(theta):
    '''positive theta value means rotate clockwise'''
    return np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])

def get_rotate(box):
    # box : x1,y2...,x3,y3
    theta = find_min_rect_angle(box)
    
#     anchor = np.array([box[::2].sum()/4,box[1::2].sum()/4])
    
    rotate_mat = get_rotate_mat(theta)
    rotated_vertices = rotate_vertices(box, theta)
    x_min, x_max, y_min, y_max = get_boundary(rotated_vertices)
    return np.array([x_min, y_min,x_max-x_min , y_max-y_min]),theta

if __name__ == '__main__':
    DATA_PATH = '/share/wuweijia/Data/ICDAR17/'
    OUT_PATH = DATA_PATH + 'annotations_coco_rotate/'
    SPLITS = ['val', 'train']
    DEBUG = False

    if not os.path.exists(OUT_PATH):
        os.mkdir(OUT_PATH)
    for split in SPLITS:
        data_path = DATA_PATH + "{}_image".format(split)
        ann_path = DATA_PATH + "{}_gt".format(split)
        image_list = os.listdir(data_path)
        print(len(image_list))
        
        out_path = OUT_PATH + '{}.json'.format(split)
        out = {'images': [], 'annotations': [], 'categories': [{'id': 1, 'name': 'person'}]}
        
        image_cnt = 0
        ann_cnt = 0
        video_cnt = 0
        for ann_data in image_list:
            image_cnt += 1
            if ann_data.split(".")[1]!="jpg":
                continue
            file_path = os.path.join(data_path,ann_data)
            im = Image.open(file_path)
            
            image_info = {'file_name': ann_data, 
                          'id': image_cnt,
                          'height': im.size[1], 
                          'width': im.size[0]}
            out['images'].append(image_info)
            
            if split != 'test':
                file_ann_path = os.path.join(ann_path,"gt_"+ann_data.replace(".jpg",".txt"))
                
                with open(file_ann_path, encoding='utf-8', mode='r') as f:
                    for line in f.readlines():
                        line = remove_all(line, '\xef\xbb\xbf')
                        gt = line.split(",")

                        points = np.array([int(gt[i]) for i in range(8)])
#                         points = np.array(object_boxes).reshape((-1))
                        points = cv2.minAreaRect(points.reshape((4, 2)))
                        points = cv2.boxPoints(points).reshape((-1))
        
                        box, rotate = get_rotate(points)
#                         points = points.reshape((4, 2))
#                         x, y, w, h = cv2.boundingRect(points)
#                         box = np.array([x, y, w, h])

                        ann_cnt += 1

                        ann = {'id': ann_cnt,
                             'category_id': 1,
                             'image_id': image_cnt,
                             'rotate': float(rotate),
                             'bbox_vis': box.tolist(),
                             'bbox': box.tolist(),
                             'area': float(box[2] * box[3]),
                             'iscrowd': 0}

                        out['annotations'].append(ann)
        print('loaded {} for {} images and {} samples'.format(split, len(out['images']), len(out['annotations'])))
        json.dump(out, open(out_path, 'w'))
