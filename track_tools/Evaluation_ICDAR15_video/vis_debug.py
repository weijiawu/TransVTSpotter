"""
author: weijia wu
"""
import os
import numpy as np
import json
import cv2
from PIL import Image, ImageDraw, ImageFont
try:
    import xml.etree.cElementTree as ET  # 解析xml的c语言版的模块
except ImportError:
    import xml.etree.ElementTree as ET

def cv2AddChineseText(img, text, position, textColor=(0, 255, 0), textSize=30):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "./track_tools/Evaluation_ICDAR13/simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text(position, text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

flag = "vis_res"  # vis_res/vis_gt


video_frames = '/share/wuweijia/Data/ICDAR2015_video/test/frames'
gt_path = './output/ICDAR15/test/gt'
res_path = './output/ICDAR15/test/best_json_tracks'
vis = './output/ICDAR15/test/vis'

if not os.path.exists(vis):
    os.makedirs(vis)

seqs = [
#     "Video_11_4_1",
        "Video_11_4_1"
]
if flag == "vis_res":
    data_path = res_path
else:
    data_path = gt_path

# 绿色: det
# 黄色：gt
# 红色：ignored
for seq in seqs:
    video_frames_ = os.path.join(video_frames,seq)
    for frame_name in os.listdir(video_frames_):
        image_path = os.path.join(video_frames_,frame_name)
        
        org_img = cv2.imread(image_path)
        
        data_path_ = os.path.join(res_path,"{}_{}.txt".format(seq,frame_name.replace(".jpg","")))
        with open(data_path_, encoding='utf-8', mode='r') as f:
            for line in f.readlines():
                params = line.strip().strip('\ufeff').strip('\xef\xbb\xbf').split(',')

                x1, y1, x2, y2, x3, y3, x4, y4 = list(map(float, params[:8]))
                bbox = [x1, y1, x2, y2, x3, y3, x4, y4]
#                 bbox = [x1,y1,x2+x1,y2,x3+x1,y3+y1,x4,y4+y1]
                bbox = np.array(bbox,np.int)
                cv2.drawContours(org_img, [bbox.reshape(int(bbox.shape[0] / 2), 2)], -1, (0, 255, 0), 2)

                
#         data_path_ = os.path.join(gt_path,"{}_{}.txt".format(seq,frame_name.replace(".jpg","")))
#         with open(data_path_, encoding='utf-8', mode='r') as f:
#             for line in f.readlines():
#                 params = line.strip().strip('\ufeff').strip('\xef\xbb\xbf').split(',')

#                 label = params[8]
# #                 if label == '*' or label == '###':
# #                     text_tags.append(True)
# #                 else:
# #                     text_tags.append(False)
#                 # if label == '*' or label == '###':
#                 x1, y1, x2, y2, x3, y3, x4, y4 = list(map(float, params[:8]))
#                 bbox = [x1, y1, x2, y2, x3, y3, x4, y4]
#                 bbox = np.array(bbox,np.int)
#                 if label == '###':
#                     cv2.drawContours(org_img, [bbox.reshape(int(bbox.shape[0] / 2), 2)], -1, (0, 0, 255), 1)
# #                     org_img=cv2AddChineseText(org_img,params[8], (int(x1), int(y1) - 20),((0, 0, 255)), 45)
#                 else:
#                     cv2.drawContours(org_img, [bbox.reshape(int(bbox.shape[0] / 2), 2)], -1, (0, 255, 255), 1)
#                     org_img=cv2AddChineseText(org_img,params[8], (int(x1), int(y1) - 20),((0, 255, 255)), 45)
                    
        cv2.imwrite(os.path.join(vis,frame_name),org_img)
