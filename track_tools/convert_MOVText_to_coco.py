"""
https://github.com/xingyizhou/CenterTrack
Modified by weijia wu
"""
import os
import numpy as np
import json
import cv2
try:
    import xml.etree.cElementTree as ET  # 解析xml的c语言版的模块
except ImportError:
    import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
import math
from tqdm import tqdm
def get_annotation(video_path):
    annotation = {}
    with open(video_path,'r',encoding='utf-8-sig') as load_f:
        gt = json.load(load_f)

    for child in gt:
        lines = gt[child]
        annotation.update({child:lines})
    return annotation

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

def get_rotate_mat(theta):
    '''positive theta value means rotate clockwise'''
    return np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])

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

def get_rotate(box):
    # box : x1,y2...,x3,y3
    theta = find_min_rect_angle(box)
    
    rotate_mat = get_rotate_mat(theta)
    rotated_vertices = rotate_vertices(box, theta)
    x_min, x_max, y_min, y_max = get_boundary(rotated_vertices)
    return np.array([x_min, y_min,x_max , y_max]),theta
    
def getBboxesAndLabels_icd13(height, width, annotations):
    bboxes = []
    labels = []
    polys = []
    bboxes_ignore = []
    labels_ignore = []
    polys_ignore = []
    IDs = []
    rotates = []
    # points_lists = [] # does not contain the ignored polygons.
    for data in annotations:
#         object_boxes = []
        object_boxes =  [int(float(i)) for i in data["points"]]
        ID = data["ID"]
        content = str(data["transcription"])
        is_caption = str(data["category"])
                    
#         for point in annotation:
#             object_boxes.append([int(point.attrib["x"]), int(point.attrib["y"])])

        points = np.array(object_boxes).reshape((-1))
        points = cv2.minAreaRect(points.reshape((4, 2)))
        # 获取矩形四个顶点，浮点型
        points = cv2.boxPoints(points).reshape((-1))
        box, rotate = get_rotate(points)

        if content == "###":
            continue
           
            
        bboxes.append(box)
        IDs.append(ID)
        rotates.append(rotate)

    if bboxes:
        bboxes = np.array(bboxes, dtype=np.float32)
        # filter the coordinates that overlap the image boundaries.
        bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, width - 1)
        bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, height - 1)
        IDs = np.array(IDs, dtype=np.int64)
        rotates = np.array(rotates, dtype=np.float32)
    else:
        bboxes = np.zeros((0, 4), dtype=np.float32)
        # polygon_point = np.zeros((0, 8), dtype=np.int)
        IDs = np.array([], dtype=np.int64)
        rotates = np.array([], dtype=np.float32)

    return bboxes, IDs, rotates

def parse_xml(annotation_path,video_path):
    
    bboxess, IDss, rotatess = [], [] , []
    annotation = get_annotation(annotation_path)
    for frame_id in annotation.keys():
        frame_name = str(frame_id) + ".jpg"
#         frame_path = os.path.join(params.split(".json")[0],frame_name)
        frame_path = os.path.join(video_path,frame_name)
        try:
            img = cv2.imread(frame_path)
            height, width = img.shape[:2]
        except:
            print(frame_path+"is None")
            continue
        bboxes, IDs, rotates = \
            getBboxesAndLabels_icd13(height, width, annotation[frame_id])   
        bboxess.append(bboxes) 
        IDss.append(IDs)
        rotatess.append(rotates)
        
    return bboxess, IDss, rotatess


def test(model,config,logger):
    model.eval()
    
    output_path = os.path.join(config.workspace, "test_output")
    input_path = os.path.join(config.target_testroot, "test_image")
    os.makedirs(output_path, exist_ok=True)
    image_list = os.listdir(input_path)
    logger.info("     ----------------------------------------------------------------")
    logger.info("                           Starting Eval...")
    logger.info("     ----------------------------------------------------------------")

    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    
    vis_path = os.path.join(config.workspace, "vis")
    if os.path.exists(vis_path):
        shutil.rmtree(vis_path)
    if not os.path.exists(vis_path):
        os.makedirs(vis_path)
        

    for one_image in tqdm(image_list):
        # one_image = "img_2.jpg"

        image_path = os.path.join(input_path, one_image)
        img = Image.open(image_path).convert('RGB')
        orign_img = cv2.imread(image_path)
        filename, file_ext = os.path.splitext(os.path.basename(one_image))
        
        res_file = output_path + "/res_" + filename + '.txt'

        boxes,score = detect_15(img, model, device)
        
#         vis_file = vis_path + "/" + filename + 'score.jpg'
#         cv2.imwrite(vis_file, score*255)

            
        with open(res_file, 'w') as f:
            if boxes is None:
                continue
            for i, box in enumerate(boxes):
                poly = np.array(box).astype(np.int32)
                points = np.reshape(poly, -1)

                strResult = ','.join(
                    [str(points[0]), str(points[1]), str(points[2]), str(points[3]), str(points[4]), str(points[5]),
                     str(points[6]), str(points[7])]) + '\r\n'

                f.write(strResult)

            if config.vis:
                for bbox in boxes:
                    # bbox = bbox / scale.repeat(int(len(bbox) / 2))
                    bbox = np.array(bbox,np.int)
                    cv2.drawContours(orign_img, [bbox[:8].reshape(int(bbox.shape[0] / 2), 2)], -1, (0, 0, 255), 2)

                vis_file = vis_path + "/" + filename + '.jpg'
                cv2.imwrite(vis_file, orign_img)

                

    f_score_new = getresult(output_path,config.gt_name)
    print("f1:",f_score_new)

def get_list(train_data_dir):
    
    img_paths = []
    gt = []
    print("Data preparing...")
    
#     train_list = os.path.join(train_data_dir,"test_list.txt")
#     image_path = os.path.join(train_data_dir,"Frames")

    with open(train_data_dir, encoding='utf-8', mode='r') as f:
        for idx,line in tqdm(enumerate(f.readlines())):
#             if idx>20:
#                 break
            
            params = line.strip().strip('\ufeff').strip('\xef\xbb\xbf')
            if ".ipy" in params:
                continue
            img_paths.append(params)    
                
#             video_path = os.path.join(os.path.join(train_data_dir,"Annotation"),params)
            
#             annotation = get_annotation(video_path)

#             for frame_id in annotation.keys():

# #                 if int(frame_id)%5!=0:
# #                     continue

# #                 frame_name = params.split("/")[1].split(".json")[0] + "_" + frame_id.zfill(6) + ".jpg"
#                 frame_name = str(frame_id) + ".jpg"
#                 frame_path = os.path.join(params.replace("GtTxtsR2Frames","Frames").split(".json")[0],frame_name)
#                 frame_path = os.path.join(image_path,frame_path)
#                 img_paths.append(frame_path)
                
#                 annotatation_frame = annotation[frame_id]

#                 bboxes = []
#                 text_tags = []
#                 for data in annotatation_frame:
#                     x1,y1,x2,y2,x3,y3,x4,y4 =  [int(float(i)) for i in data["points"]]
#                     ID = data["ID"]
#                     content = str(data["transcription"])
#                     is_caption = str(data["category"])
                    

#                     box = np.array([x1, y1, x2, y2, x3, y3, x4, y4], np.float)
                    
#                     bboxes.append(box)
#                     text_tags.append(False)

#                 gt.append(bboxes)

    return img_paths

if __name__ == '__main__':
    
    # Use the same script for MOT16
    DATA_PATH = '/share/wuweijia/MyBenchMark/relabel/Dapan_lizhuang/final_FrameAnn/MOVText'
    OUT_PATH = os.path.join(DATA_PATH, 'annotations_coco_rotate')
    # SPLITS = ['train_half', 'val_half', 'train','test']  # --> split training data to train_half and val_half.
    SPLITS = ["test"]

    HALF_VIDEO = True
    CREATE_SPLITTED_ANN = True
    CREATE_SPLITTED_DET = True

    if not os.path.exists(OUT_PATH):
        os.makedirs(OUT_PATH)

    for split in SPLITS:
        if split == "test":
            data_path = os.path.join(DATA_PATH, 'Frames')
            ann_path_ = os.path.join(DATA_PATH, 'Annotation')
            seq_list = get_list(os.path.join(DATA_PATH,"test_list.txt"))
            
        else:
            data_path = os.path.join(DATA_PATH, 'Frames')
            ann_path_ = os.path.join(DATA_PATH, 'Annotation')
            seq_list = get_list(os.path.join(DATA_PATH,"train_list.txt"))
            
        out_path = os.path.join(OUT_PATH, '{}.json'.format(split))
        out = {'images': [], 'annotations': [], 'videos': [],
               'categories': [{'id': 1, 'name': 'pedestrian'}]}
#         seqs = os.listdir(data_path)
        seqs = seq_list
        image_cnt = 0
        ann_cnt = 0
        video_cnt = 0
        for seq in sorted(seqs):
            if '.DS_Store' in seq:
                continue
#             if 'mot' in DATA_PATH and (split != 'test' and not ('FRCNN' in seq)):
#                 continue
            video_cnt += 1  # video sequence number.
            print(seq.replace("/","_"))
            out['videos'].append({'id': video_cnt, 'file_name': seq.replace("/","_")})
            seq_path = os.path.join(data_path, seq)
            img_path = seq_path.split(".jso")[0]
            ann_path = os.path.join(ann_path_, seq)
            images = os.listdir(img_path)
            num_images = len([image for image in images if 'jpg' in image])  # half and half

            if HALF_VIDEO and ('half' in split):
                image_range = [0, num_images // 2] if 'train' in split else \
                              [num_images // 2 + 1, num_images - 1]
            else:
                image_range = [0, num_images - 1]

            for i in range(num_images):
                if i < image_range[0] or i > image_range[1]:
                    continue
#                 print(os.path.join(img_path, '{}/{}.jpg'.format(seq, i + 1)))
#                 img = cv2.imread(os.path.join(img_path, '{}.jpg'.format(i + 1)))
#                 print(os.path.join(data_path, '{}/{:06d}.jpg'.format(seq, i + 1)))
                height, width = 0,0
                image_info = {'file_name': '{}/{}.jpg'.format(seq.split(".jso")[0], i + 1),  # image name.
                              'id': image_cnt + i + 1,  # image number in the entire training set.
                              'frame_id': i + 1 - image_range[0],  # image number in the video sequence, starting from 1.
                              'prev_image_id': image_cnt + i if i > 0 else -1,  # image number in the entire training set.
                              'next_image_id': image_cnt + i + 2 if i < num_images - 1 else -1,
                              'video_id': video_cnt,
                              'height': height, 'width': width}
                out['images'].append(image_info)
            print('{}: {} images'.format(seq, num_images))
            
            
            if split != 'test':
                bboxess, IDss, rotatess = parse_xml(ann_path,img_path)
#                 seq_path_ = os.path.join(OUT_PATH, seq)
#                 if not os.path.exists(seq_path_):
#                     os.makedirs(seq_path_)
        
#                 gt_out = os.path.join(seq_path_, 'gt_{}.txt'.format(split))
#                 fout = open(gt_out, 'w')
                
                print('{} ann images'.format(len(IDss)))
                for i in range(len(IDss)):
                    frame_id = i + 1
                    if frame_id - 1 < image_range[0] or frame_id - 1 > image_range[1]:
                        continue
                    category_id = 1
                    
                    for bboxes,IDs,rotates in zip(bboxess[i],IDss[i],rotatess[i]):
                        track_id = int(IDs)
                        bboxes[2] = bboxes[2]-bboxes[0]
                        bboxes[3] = bboxes[3]-bboxes[1]
                        ann_cnt += 1
                        ann = {'id': ann_cnt,
                               'category_id': category_id,
                               'image_id': image_cnt + frame_id,
                               'track_id': track_id,
                               'rotate': float(rotates),
                               'bbox': bboxes.tolist(),
                               'conf': 1.0,
                               'iscrowd': 0,
                               'area': float(bboxes[2] * bboxes[3])}
                        out['annotations'].append(ann)
                        
                        
#                         o = [frame_id-image_range[0],track_id,bboxes[0],bboxes[1],bboxes[2],bboxes[3],1,1,1]
                        

#                         fout.write('{:d},{:d},{:d},{:d},{:d},{:d},{:d},{:d},{:.6f}\n'.format(
#                                     int(o[0]), int(o[1]), int(o[2]), int(o[3]), int(o[4]), int(o[5]),
#                                     int(o[6]), int(o[7]), o[8]))
#                 fout.close()
                    
            image_cnt += num_images
        print('loaded {} for {} images and {} samples'.format(split, len(out['images']), len(out['annotations'])))
        json.dump(out, open(out_path, 'w'))
        
