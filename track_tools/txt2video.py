import os
import sys
import json
import cv2
import glob as gb
from track_tools.colormap import colormap
from tqdm import tqdm
import moviepy
import moviepy.video.io.ImageSequenceClip
from moviepy.editor import *
import shutil

def txt2img(visual_path="visual_val_gt",data = "icdar2015"):
    print("Starting txt2img")

    valid_labels = {1}
    ignore_labels = {2, 7, 8, 12}

    if not os.path.exists(visual_path):
        os.makedirs(visual_path)
    color_list = colormap()
    
    if data == "icdar2015":
        gt_json_path = '/share/wuweijia/Data/ICDAR2013_video/annotations_coco_rotate/test.json'
        img_path = '/share/wuweijia/Data/ICDAR2013_video/test/frames/'
    else:
        gt_json_path = '/share/wuweijia/Data/ICDAR2013_video/annotations_coco_rotate/test.json'
        img_path = '/share/wuweijia/Data/ICDAR2013_video/test/frames/'
    
    show_video_names = [
#         'Video_11_4_1'
        'Video_5_3_2'
        
                       ]


    test_json_path = 'mot/annotations/test.json'
    test_img_path = 'mot/test/'
    test_show_video_names = ['MOT20-01', 
                    'MOT20-02',
                    'MOT20-03',
                    'MOT20-05']
    if visual_path == "visual_test_predict":
        show_video_names = test_show_video_names
        img_path = test_img_path
        gt_json_path = test_json_path
        
    for show_video_name in show_video_names:
        img_dict = dict()
        
        if visual_path == "visual_val_gt":
            txt_path = '/share/wuweijia/Data/ICDAR2013_video/annotations_coco_rotate/' + show_video_name + '/gt_train.txt'
            
        elif visual_path == "visual_val_predict":
            if data == "icdar2015":
                txt_path = 'output/ICDAR15/test/tracks/'+ show_video_name + '.txt'
            else:
                txt_path = 'output/ICDAR13/test/tracks/'+ show_video_name + '.txt'
            
        elif visual_path == "visual_test_predict":
            txt_path = 'test/tracks/'+ show_video_name + '.txt'
        else:
            raise NotImplementedError
        
        with open(gt_json_path, 'r') as f:
            gt_json = json.load(f)

        for ann in gt_json["images"]:
            file_name = ann['file_name']
            video_name = file_name.split('/')[0]
            if video_name == show_video_name:
                print(ann['frame_id'])
                img_dict[ann['frame_id']] = img_path + file_name


        txt_dict = dict()    
        with open(txt_path, 'r') as f:
            for line in f.readlines():
                linelist = line.split(',')

                mark = int(float(linelist[6]))
                label = int(float(linelist[7]))
                vis_ratio = float(linelist[8])
                
                if visual_path == "visual_val_gt":
                    if mark == 0 or label not in valid_labels or label in ignore_labels or vis_ratio <= 0:
                        continue

                img_id = linelist[0]
                obj_id = linelist[1]
                bbox = [float(linelist[2]), float(linelist[3]), 
                        float(linelist[2]) + float(linelist[4]), 
                        float(linelist[3]) + float(linelist[5]), int(obj_id)]
                if int(img_id) in txt_dict:
                    txt_dict[int(img_id)].append(bbox)
                else:
                    txt_dict[int(img_id)] = list()
                    txt_dict[int(img_id)].append(bbox)

        for img_id in sorted(txt_dict.keys()):
#             print(img_dict[img_id])
            img = cv2.imread(img_dict[img_id])
            for bbox in txt_dict[img_id]:
                cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color_list[bbox[4]%79].tolist(), thickness=2)
                cv2.putText(img, "{}".format(int(bbox[4])), (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_list[bbox[4]%79].tolist(), 2)
            cv2.imwrite(visual_path + "/" + show_video_name + "{:0>6d}.png".format(img_id), img)
        print(show_video_name, "Done")
    print("txt2img Done")

        
def img2video(visual_path="visual_val_gt"):
    print("Starting img2video")

    img_paths = gb.glob(visual_path + "/*.png") 
    fps = 16 
    size = (1920,1080) 
    videowriter = cv2.VideoWriter(visual_path + "_video.avi",cv2.VideoWriter_fourcc('M','J','P','G'), fps, size)

    for img_path in sorted(img_paths):
        img = cv2.imread(img_path)
        img = cv2.resize(img, size)
        videowriter.write(img)

    videowriter.release()
    print("img2video Done")

def pics2video(frames_dir="", fps=20):
#     im_names = os.listdir(frames_dir)
    im_names = gb.glob(frames_dir + "/*.png") 
    
    num_frames = len(im_names)
    frames_path = []
    for img_path in sorted(im_names):
#         string = os.path.join( frames_dir, str(im_name) + '.jpg')
        frames_path.append(img_path)
#     for im_name in tqdm(range(1, num_frames+1)):
#         string = os.path.join( frames_dir, str(im_name) + '.jpg')
#         frames_path.append(string)
        
#     frames_name = sorted(os.listdir(frames_dir))
#     frames_path = [frames_dir+frame_name for frame_name in frames_name]
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(frames_path, fps=fps)
    clip.write_videofile(frames_dir+".mp4", codec='libx264')
    shutil.rmtree(frames_dir)
    
if __name__ == '__main__':
#     'visual_val_gt'  "visual_val_predict"
    visual_path="visual_val_predict"
    if len(sys.argv) > 1:
        visual_path =sys.argv[1]
    txt2img(visual_path)
#     img2video(visual_path)
    pics2video(visual_path)
