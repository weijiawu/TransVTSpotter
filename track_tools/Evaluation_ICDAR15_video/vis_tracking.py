# -*- coding: utf-8 -*-
import cv2
import os
import copy
import numpy as np
import math
# import Levenshtein
from cv2 import VideoWriter, VideoWriter_fourcc
import json
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import moviepy
import moviepy.video.io.ImageSequenceClip
import shutil
from moviepy.editor import *

def pics2video(frames_dir="", fps=25):
    im_names = os.listdir(frames_dir)
    num_frames = len(im_names)
    frames_path = []
    for im_name in tqdm(range(1, num_frames+1)):
        string = os.path.join( frames_dir, str(im_name) + '.jpg')
        frames_path.append(string)
        
#     frames_name = sorted(os.listdir(frames_dir))
#     frames_path = [frames_dir+frame_name for frame_name in frames_name]
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(frames_path, fps=fps)
    clip.write_videofile(frames_dir+".mp4", codec='libx264')
    shutil.rmtree(frames_dir)
    
def Frames2Video(frames_dir=""):
    '''  将frames_dir下面的所有视频帧合成一个视频 '''
    img_root = frames_dir      #'E:\\KSText\\videos_frames\\video_14_6'
    image = cv2.imread(os.path.join(img_root,"1.jpg"))
    h,w,_ = image.shape

    out_root = frames_dir+".avi"
    # Edit each frame's appearing time!
    fps = 20
    fourcc = VideoWriter_fourcc(*"MJPG")  # 支持jpg
    videoWriter = cv2.VideoWriter(out_root, fourcc, fps, (w, h))
    im_names = os.listdir(img_root)
    num_frames = len(im_names)
    print(len(im_names))
    for im_name in tqdm(range(1, num_frames+1)):
        string = os.path.join( img_root, str(im_name) + '.jpg')
#         print(string)
        frame = cv2.imread(string)
        # frame = cv2.resize(frame, (w, h))
        videoWriter.write(frame)

    videoWriter.release()
    shutil.rmtree(img_root)
    
def get_annotation(video_path):
    annotation = {}

    

    with open(video_path,'r',encoding='utf-8-sig') as load_f:
        gt = json.load(load_f)

    for child in gt:
        lines = gt[child]
        annotation.update({child:lines})

    return annotation

def cv2AddChineseText(img, text, position, textColor=(0, 255, 0), textSize=30):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "./track_tools/Evaluation_ICDAR15_video/simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text(position, text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

def mask_image(image, mask_2d, rgb=None, valid = False):
    h, w = mask_2d.shape

    # mask_3d = np.ones((h, w), dtype="uint8") * 255
    mask_3d_color = np.zeros((h, w, 3), dtype="uint8")
    # mask_3d[mask_2d[:, :] == 1] = 0
    
        
    image.astype("uint8")
    mask = (mask_2d!=0).astype(bool)
    if rgb is None:
        rgb = np.random.randint(0, 255, (1, 3), dtype=np.uint8)
        
    mask_3d_color[mask_2d[:, :] == 1] = rgb
    image[mask] = image[mask] * 0.5 + mask_3d_color[mask] * 0.5
    
    if valid:
        mask_3d_color[mask_2d[:, :] == 1] = [[0,0,0]]
        kernel = np.ones((5,5),np.uint8)  
        mask_2d = cv2.dilate(mask_2d,kernel,iterations = 4)
        mask = (mask_2d!=0).astype(bool)
        image[mask] = image[mask] * 0 + mask_3d_color[mask] * 1
        return image,rgb
        
    return image,rgb

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

if __name__ == "__main__":
# ['Video_49_6_4', 'Video_43_6_4', 'Video_38_2_3', 'Video_50_7_4', 'Video_5_3_2', 'Video_55_3_2', 'Video_23_5_2', 'Video_44_6_4', 'Video_22_5_1', 'Video_15_4_1', 'Video_39_2_3', 'Video_24_5_2', 'Video_9_1_1', 'Video_20_5_1', 'Video_35_2_3', 'Video_48_6_4', 'Video_30_2_3', 'Video_1_1_2', 'Video_17_3_1', 'Video_34_2_3', 'Video_11_4_1', 'Video_53_7_4', 'Video_6_3_2', 'Video_32_2_3']
    
#     root = "/share/wuweijia/MyBenchMark/MMVText/MMVText_30S/MMVText_30s/"
    annotation_path = "./output/ICDAR15/test/best_json_tracks"
#     video_path = root + "Video"
    frame_path = '/share/wuweijia/Data/ICDAR2015_video/test/frames'
    result_path_cls = "./output/ICDAR15/vis"
    
    gt_ann_path = "/share/wuweijia/Code/VideoSpotting/TransSpotter/track_tools/Evaluation_ICDAR13/Eval_Tracking/gt"
    
    # Video_11_4_1.json
    # Video_5_3_2
    seqs_video = ["Video_1_1_2"]
    
    if not os.path.exists(result_path_cls):
        os.makedirs(result_path_cls)
    
#     print(os.listdir(frame_path))
#     assert False
    for video in tqdm(os.listdir(frame_path)):
            
        if video.split(".mp4")[0] not in seqs_video:
            continue

        annotation_path_cls_v = os.path.join(annotation_path, video+".json").replace("V","res_v").replace("_1_2","")
        gt_ann = os.path.join(gt_ann_path, video+"_GT.json")
#         video_path_cls_v = os.path.join(video_path_cls, video)
        frame_path_cls_v = os.path.join(frame_path, video.split(".mp4")[0])

#         video_1 = VideoFileClip(video_path_cls_v)
#         fps = video_1.fps

        annotation = get_annotation(annotation_path_cls_v)
        gt_ann_ = get_annotation(gt_ann)

        result_path_cls_video = os.path.join(result_path_cls, video.split(".mp4")[0])
        if not os.path.exists(result_path_cls_video):
            os.makedirs(result_path_cls_video)
#             gap = int(fps/5)
        gap=1
        lis = np.arange(0,100000,gap)+1
        rgbs={}
        for idx,frame_id in tqdm(enumerate(annotation.keys())):
#             print(frame_id)

            if int(frame_id) in lis:
                frame_id_ = frame_id
                frame_id_im = frame_id
            else:
                frame_id_ = frame_id
                frame_id_im = frame_id


            while int(frame_id_) not in lis:
                frame_id_ = str(int(frame_id_)-1)
                frame_id_im = frame_id_


#             frame_name = video_name.split(".json")[0] + "_" + frame_id_im.zfill(6) + ".jpg"
            frame_name = frame_id_im + ".jpg"
            frame_path = os.path.join(frame_path_cls_v,frame_name)
            frame = cv2.imread(frame_path)

            annotatation_frame = annotation[frame_id_]
            annotatation_frame_gt = gt_ann_[frame_id_]
            for data in annotatation_frame_gt:
                x1,y1,x2,y2,x3,y3,x4,y4 = data["points"]
                ID = "   "+data["ID"]
                tans = data["transcription"]
#                 print(trans)
                points = np.array([x1,y1,x2,y2,x3,y3,x4,y4]).reshape((-1))
                points = cv2.minAreaRect(points.reshape((4, 2)))
                # 获取矩形四个顶点，浮点型
                x1, y1, x2, y2, x3, y3, x4, y4 = cv2.boxPoints(points).reshape((-1))
                
                x1,y1,x2,y2,x3,y3,x4,y4 = adjust_box_sort(np.array([x1, y1, x2, y2, x3, y3, x4, y4], np.int32))

                if tans == "###":
                    points = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], np.int32)
                    cv2.polylines(frame, [points], True, (0,0,255), thickness=3)
                    frame=cv2AddChineseText(frame,str(tans), (int(x1), int(y1) - 20),((0,0,255)), 30)
                else:
                    points = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], np.int32)
                    cv2.polylines(frame, [points], True, (0,255,0), thickness=3)
                    frame=cv2AddChineseText(frame,str(tans), (int(x1), int(y1) - 20),((0,255,0)), 30)
                
            for data in annotatation_frame:
#                 x1,y1,x2,y2,x3,y3,x4,y4,ID, content,is_caption = data
                x1,y1,x2,y2,x3,y3,x4,y4 = data["points"]
                ID = data["ID"]
                tans = data["transcription"]
#                 id_content = str(data["transcription"])+ ","+ str(ID)
#                 is_caption = str(data["category"])

                points = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], np.int32)
                mask_1 = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                cv2.fillPoly(mask_1, [points], 1)

                if ID in rgbs:
                    frame,rgb = mask_image(frame, mask_1,rgbs[ID])
                else:
                    frame,rgb = mask_image(frame, mask_1)
                    rgbs[ID] = rgb

                r,g,b = rgb[0]
                r,g,b = int(r),int(g),int(b)
#                 cv2.polylines(frame, [points], True, (r,g,b), thickness=3)
#                 if is_caption == "scene":
                points = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], np.int32)
                cv2.polylines(frame, [points], True, (r,g,b), thickness=3)
#                 frame=cv2AddChineseText(frame,tans, (int(x1), int(y1) - 20),((r,g,b)), 30)
#                 else:
#                     points = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], np.int32)
#                     cv2.polylines(frame, [points], True, (r,g,b), thickness=5)
#                     frame=cv2AddChineseText(frame,id_content, (int(x1), int(y1) - 20),((r,g,b)), 45)

            frame_vis_path = os.path.join(result_path_cls_video, frame_id+".jpg")
            cv2.imwrite(frame_vis_path, frame)
#             video_vis_path = "./"
        pics2video(result_path_cls_video,fps=5)
#             break
#         break




        
        