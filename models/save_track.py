"""
Copyright (c) https://github.com/xingyizhou/CenterTrack
Modified by weijiawu
"""
# coding: utf-8
import os
import json
import logging
from collections import defaultdict
import numpy as np
import math
from cv2 import VideoWriter, VideoWriter_fourcc
from tqdm import tqdm
from collections import OrderedDict
from shapely.geometry import Polygon, MultiPoint
try:
    import xml.etree.cElementTree as ET  #解析xml的c语言版的模块
except ImportError:
    import xml.etree.ElementTree as ET
from xml.dom.minidom import Document

class StorageDictionary(object):
    @staticmethod
    def dict2file(file_name, data_dict):
        try:
            import cPickle as pickle
        except ImportError:
            import pickle
        # import pickle
        output = open(file_name,'wb')
        pickle.dump(data_dict,output)
        output.close()

    @staticmethod
    def file2dict(file_name):
        try:
            import cPickle as pickle
        except ImportError:
            import pickle
        # import pickle
        pkl_file = open(file_name, 'rb')
        data_dict = pickle.load(pkl_file)
        pkl_file.close()
        return data_dict

    #Python语言特定的序列化模块是pickle，但如果要把序列化搞得更通用、更符合Web标准，就可以使用json模块
    @staticmethod
    def dict2file_json(file_name, data_dict):
        import json, io
        with io.open(file_name, 'w', encoding='utf-8') as fp:
            # fp.write(unicode(json.dumps(data_dict, ensure_ascii=False, indent=4) ) )  #可以解决在文件里显示中文的问题，不加的话是 '\uxxxx\uxxxx'
            fp.write((json.dumps(data_dict, ensure_ascii=False, indent=4) ) )


    @staticmethod
    def file2dict_json(file_name):
        import json, io
        with io.open(file_name, 'r', encoding='utf-8') as fp:
            data_dict = json.load(fp)
        return data_dict

# 普通 dict 插入元素时是无序的，使用 OrderedDict 按元素插入顺序排序
# 对字典按key排序, 默认升序, 返回 OrderedDict
def sort_key(old_dict, reverse=False):
    """对字典按key排序, 默认升序, 不修改原先字典"""
    # 先获得排序后的key列表
    keys = [int(i) for i in old_dict.keys()]
    keys = sorted(keys, reverse=reverse)
    # 创建一个新的空字典
    new_dict = OrderedDict()
    # 遍历 key 列表
    for key in keys:
        new_dict[str(key)] = old_dict[str(key)]
    return new_dict

def Generate_Json_annotation(TL_Cluster_Video_dict, Outpu_dir,xml_dir_):
    '''   '''
    ICDAR21_DetectionTracks = {}
    text_id = 1
    
    doc = Document()
    video_xml = doc.createElement("Frames")
    
    for frame in TL_Cluster_Video_dict.keys():
        
        doc.appendChild(video_xml)
        aperson = doc.createElement("frame")
        aperson.setAttribute("ID", str(frame))
        video_xml.appendChild(aperson)

        ICDAR21_DetectionTracks[frame] = []
        for text_list in TL_Cluster_Video_dict[frame]:
            ICDAR21_DetectionTracks[frame].append({"points":text_list[:8],"ID":text_list[8]})
            
            # xml
            object1 = doc.createElement("object")
            object1.setAttribute("ID", str(text_list[8]))
            aperson.appendChild(object1)
            
            for i in range(4):
                
                name = doc.createElement("Point")
                object1.appendChild(name)
                # personname = doc.createTextNode("1")
                name.setAttribute("x", str(int(text_list[i*2])))
                name.setAttribute("y", str(int(text_list[i*2+1])))
        
            
    StorageDictionary.dict2file_json(Outpu_dir, ICDAR21_DetectionTracks)
    
    # xml
    f = open(xml_dir_, "w")
    f.write(doc.toprettyxml(indent="  "))
    f.close()
    
    
def get_rotate_mat(theta):
	'''positive theta value means rotate clockwise'''
	return np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])

def save_track(results, out_root, video_to_images, video_names, data_split='val',det_val=False):
    assert out_root is not None
    out_dir = os.path.join(out_root, data_split)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # save json.
    json_path = os.path.join(out_dir, "track_results.json")
    with open(json_path, "w") as f:
        f.write(json.dumps(results))
        f.flush()

    # save it in standard format.
    track_dir = os.path.join(out_dir, "tracks")
    xml_dir = os.path.join(out_dir, "xml_dir")
    eval_dir = os.path.join(out_dir, "res")
#     json_root = os.path.join(out_dir, "track")

    if not os.path.exists(track_dir):
        os.mkdir(track_dir)
    if not os.path.exists(eval_dir):
        os.mkdir(eval_dir)
    if not os.path.exists(xml_dir):
        os.mkdir(xml_dir)
        
    for video_id in video_to_images.keys():
        video_infos = video_to_images[video_id]
        video_name = video_names[video_id]
        print(video_name)
        file_path = os.path.join(track_dir, "{}.txt".format(video_name))
        
        tracks = defaultdict(list)
        annotation = {}
        output_cls = os.path.join(track_dir,"{}.json".format(video_name))
        xml_name = video_name.split("_")
        
        xml_name = xml_name[0] + "_" + xml_name[1]
#         xml_name = xml_name[0]
        
        xml_dir_ = os.path.join(xml_dir,"res_{}.xml".format(xml_name.replace("V","v")))
        for video_info in tqdm(video_infos):
            image_id, frame_id = video_info["image_id"], video_info["frame_id"]
            result = results[image_id]
            
            boxes_list = []
            box = []
#             print("frame_id",frame_id,len(result))   
            for item in result:
                if not ("tracking_id" in item):
                    raise NotImplementedError
                tracking_id = item["tracking_id"]
                rotate = item["rotate"]
                bbox = item["bbox"]
   
                x_min,y_min, x_max, y_max = bbox
                rotate_mat = get_rotate_mat(-rotate)
                temp_x = np.array([[x_min, x_max, x_max, x_min]]) - (x_min+x_max)/2
                temp_y = np.array([[y_min, y_min, y_max, y_max]]) - (y_min+y_max)/2
                coordidates = np.concatenate((temp_x, temp_y), axis=0)
                res = np.dot(rotate_mat, coordidates)
                res[0,:] += (x_min+x_max)/2
                res[1,:] += (y_min+y_max)/2
                detetcion_box = [res[0,0], res[1,0], res[0,1], res[1,1], res[0,2], res[1,2],res[0,3], res[1,3]]
                track_box = [res[0,0], res[1,0], res[0,1], res[1,1], res[0,2], res[1,2],res[0,3], res[1,3],tracking_id]

#                 detetcion_box = [int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[1]),
#                                             int(bbox[2]),int(bbox[3]),int(bbox[0]),int(bbox[3])]
                
    
                bbox = [bbox[0], bbox[1], bbox[2], bbox[3], item['score'], item['active']]
                tracks[tracking_id].append([frame_id] + bbox)
                
                if item['active']>0:
                    boxes_list.append(detetcion_box)
                    box.append(track_box)

            
#             txt_path = os.path.join(eval_dir,"{}_{}.txt".format(video_name,frame_id))
#             np.savetxt(txt_path, np.array(boxes_list).reshape(-1, 8), delimiter=',', fmt='%d')
        
            annotation.update({str(frame_id):box}) 
        annotation = sort_key(annotation)
        Generate_Json_annotation(annotation,output_cls,xml_dir_)
            

    
    