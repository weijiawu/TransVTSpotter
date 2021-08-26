"""
https://github.com/xingyizhou/CenterTrack
Modified by weijia wu
"""
import os
import numpy as np
import json
import cv2
import shutil

try:
    import xml.etree.cElementTree as ET  # 解析xml的c语言版的模块
except ImportError:
    import xml.etree.ElementTree as ET
from util.utils import write_result_as_txt,debug, setup_logger,write_lines,MyEncoder
from tqdm import tqdm

# Use the same script for MOT16
DATA_PATH = '/share/wuweijia/Data/ICDAR2015_video'
OUT_PATH = "./output/ICDAR15/test/gt"
submit = "/share/wuweijia/Code/VideoSpotting/TransTrack/output/ICDAR13/test/evaluation/"


def getBboxesAndLabels_icd13(height, width, annotations):
    bboxes = []
    labels = []
    polys = []
    bboxes_ignore = []
    labels_ignore = []
    polys_ignore = []
    IDs = []
    # points_lists = [] # does not contain the ignored polygons.
    for annotation in annotations:
        object_boxes = []
        for point in annotation:
            object_boxes.append([int(point.attrib["x"]), int(point.attrib["y"])])
        points = np.array(object_boxes).reshape((-1))
        points = cv2.minAreaRect(points.reshape((4, 2)))
        # 获取矩形四个顶点，浮点型
        points = cv2.boxPoints(points).reshape((-1))
        
#         points = np.array(object_boxes)
#         x, y, w, h = cv2.boundingRect(points)
#         box = np.array([x, y, x+w, y, x+w, y+h, x, y+h])
#         box[0::2] = np.clip(box[0::2], 0, width - 1)
#         box[1::2] = np.clip(box[1::2], 0, height - 1)
        
        line = str(box[0])
        for b in range(1,len(box)):
            line += ","
            line += str(box[b])

        Transcription = annotation.attrib["Transcription"]
        if "?" in Transcription or "#" in Transcription or "55" in Transcription:
            line += ","
            line += "###"
        else:
            line += ","
            line += Transcription
           
        line += "\n"
        
        bboxes.append(line)


    return bboxes

def parse_xml(annotation_path,video_path,gt_path):
    utf8_parser = ET.XMLParser(encoding='gbk')
    with open(annotation_path, 'r', encoding='gbk') as load_f:
        tree = ET.parse(load_f, parser=utf8_parser)
    root = tree.getroot()  # 获取树型结构的根
    
    for idx,child in enumerate(root):
        image_path = os.path.join(video_path, child.attrib["ID"] + ".jpg")
        try:
            img = cv2.imread(image_path)
            height, width = img.shape[:2]
        except:
            print(image_path+"is None")
            continue
        bboxes = \
            getBboxesAndLabels_icd13(height, width, child)
        
        gt_path_txt = os.path.join(gt_path,"{}_{}.txt".format(video_path.split("/")[-1],child.attrib["ID"]))
        write_lines(gt_path_txt, bboxes) 
    

if __name__ == '__main__':

    if not os.path.exists(OUT_PATH):
        os.makedirs(OUT_PATH)
        
    data_path = os.path.join(DATA_PATH, 'test/frames')
    ann_path_ = os.path.join(DATA_PATH, 'test/gt')
    seqs = os.listdir(data_path)

    for seq in tqdm(sorted(seqs)):
        ann_path = os.path.join(ann_path_, seq + "_GT_voc.xml")
        parse_xml(ann_path,os.path.join(data_path,seq),OUT_PATH)
    
#     for i in os.listdir(submit):
#         if i not in os.listdir(OUT_PATH):
#             shutil.rmtree(os.path.join(submit,i))
        