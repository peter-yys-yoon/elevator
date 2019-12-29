"------------------------PYTHON-------------------------------"
import os
import cv2
from opt import opt
from tqdm import tqdm
from pathlib import Path
import time

from vars import *
from bgtest import BGsubs


from tiah.tools import read_video, get_framesize
from tiah.vars import *
from tiah.get_data import get_bbox_dict
from tiah.Drawing import draw_header
from tiah.pose import *

def get_dy_list():
    path = f'data/dyselect/good'

    flist = os.listdir(path)
    flist.sort()
    fflist = []
    for f in flist:
        ff = f.split('_')[1]
        fflist.append(os.path.join(PATH_DATA_VIDEO,ff))
    return fflist


def video_reading():
    args = opt
    WAIT = 1

    dylist =get_dy_list()
    # args.video =dylist[-1]
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
    while 1:
        ret, frame = cap.read()
        if ret is False:
            break

        vis_frame = frame.copy()
        
        
        if count in bbox_list_dict.keys():
            bbox_list = bbox_list_dict[count]
        else:
            bbox_list = []
            
        if count in pose_list_dict.keys():
            pose_list = pose_list_dict[count]
        else:
            pose_list = []

        for bbox in bbox_list: # drawing bbox
            idx, x1, y1, x2, y2, conf, cls_conf, cls_pred = bbox
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), COLOR_GREEN, 1)


        "-------------------------MAIN MODULE----------------------"
        # vis_frame = bgmodel.apply(vis_frame, count, bbox_list)
        
        draw_pose(vis_frame, pose_list)


        "-----------------------------------------------------------"


        draw_header(vis_frame, count, COLOR_BLACK, scale=1,thick=1 )
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
