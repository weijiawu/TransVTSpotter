# -*- coding: utf-8 -*-
import cv2
import os
import os.path as osp

root = '/share/wuweijia/Data/ICDAR2013_video/test/'  # dir of videos

video_path = root + "video"
save_path = root + "frames"

if not osp.exists(save_path):
    os.mkdir(save_path)

file_list=os.listdir(video_path)
for f in file_list:
    save_dir = osp.join(save_path, f.split('.')[0])
    if not osp.exists(save_dir):
        os.mkdir(save_dir)
    
#     print(osp.join(root, f))
    # input video names
    videoCapture = cv2.VideoCapture(osp.join(video_path, f))  # ./videos/video_17_3.avi

    # get video property info
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    print('fps: {}'.format(fps))
    print('size: {}'.format(size))

    # parsing to frames
    frame_num = 0
    success, frame = videoCapture.read()
    while success:
        frame_num += 1
        cv2.imwrite(osp.join(save_dir, str(frame_num) + '.jpg'), frame)
        success, frame = videoCapture.read() #获取下一帧

    print('a total of {} frames..'.format(frame_num))