"------------------------PYTHON-------------------------------"
import os
import cv2
from opt import opt
from tqdm import tqdm
from pathlib import Path
import time
from random import randrange

from vars import *
from detections import BGsubs, FallDetection

from tiah.tools import read_video, get_framesize
from tiah.vars import *
from tiah.get_data import get_bbox_dict
from tiah.Drawing import draw_header
from tiah.pose import *

"""--------------------------------------------"
A001: fight
A002: Fall
A003: Panic
A004: Paper
A005: Escape
A006: Standing
"--------------------------------------------"""

preset_mm = randrange(1,46)
cam_id = randrange(6)


def get_dy_list():
    path = f'data/dyselect/good'

    flist = os.listdir(path)
    flist.sort()
    fflist = []
    for f in flist:
        ff = f.split('_')[1]
        fflist.append(os.path.join(PATH_DATA_VIDEO,ff))
    return fflist

def get_time_string(count):
    
    time_in_sec = count//25
    mm , ss = divmod(time_in_sec, 60)
    hh =17 

    return f'{hh}:{mm+preset_mm}:{str(ss).zfill(2)}'


def get_header_label(count, bbox_list):
    
    time_string = get_time_string(count)
    person_string = len(bbox_list)
    return f'Cam {cam_id}  {time_string}  Passenger {person_string}'

def filter_list(bbox_list):
    person_list = []
    for bbox in bbox_list:
        idx, x1, y1, x2, y2, conf, cls_conf, cls_pred = bbox
        if cls_pred == 'person':
            if conf > 0.5:
                person_list.append(bbox)

    return person_list
 

def get_list(list_dict, count):
    if count in list_dict.keys():
        return list_dict[count]
    else:
        return []

def video_reading():
    args = opt
    WAIT = 1


    # dylist =get_dy_list()
    # args.video =dylist[-1]
    args.video = os.path.join(PATH_DATA_VIDEO,'C001A004D001P0003T0002.avi')
    "----------------------------------"
    cap, atts = read_video(opt.video)
    framesize = get_framesize(cap)

    bbox_txt_path = os.path.join(PATH_DATA_YOLOTXT, f'yolo_{Path(args.video).stem}.txt')
    bbox_list_dict = get_bbox_dict(bbox_txt_path)
    pose_json_path = os.path.join(PATH_DATA_JSONJSON, f'alpha_{Path(args.video).stem}.json')
    pose_list_dict = get_pose_list_from_json(pose_json_path)
    
    if args.save_video:
        savepath = f'{args.outdir}/{Path(args.video).stem}.avi'
        writer = cv2.VideoWriter(savepath, H264, 20, framesize)

    name_desc = tqdm(range(atts[LENGTH]))
    count = 0
    bgmodel = BGsubs()
    fallmodel = FallDetection(FALL_THRESHOLD)

    while 1:
        ret, frame = cap.read()
        if ret is False:
            break

        vis_frame = frame.copy()
        
        bbox_list = get_list(bbox_list_dict,count)
        person_list= filter_list(bbox_list)
        pose_list= get_list(pose_list_dict,count)
        
        for bbox in person_list: # drawing bbox
            idx, x1, y1, x2, y2, conf, cls_conf, cls_pred = bbox
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), COLOR_GREEN, 1)


        "-------------------------MAIN MODULE----------------------"
        vis_frame = bgmodel.apply(vis_frame, count, person_list)
        
        # draw_pose(vis_frame, pose_list)

        color =COLOR_BLACK
        action = ''
        # if fallmodel.detect(bbox_list):
        #     color = COLOR_ORANGE
        #     msg ='Fall-Down'
        

        "-----------------------------------------------------------"
        header_label = get_header_label(count, person_list)
        aa = f'{header_label} {action}'
        print(aa)
        draw_header(vis_frame, count, color, msg = aa, scale=1,thick=1 )
        if args.save_video:
            writer.write(vis_frame)
        if args.vis:
            cv2.imshow('11', vis_frame)
            if cv2.waitKey(WAIT) == ord('q'):
                break

        name_desc.update(1)
        
        count += 1

    if args.save_video:
        writer.release()


if __name__ == "__main__":
    video_reading()
