import os
import cv2
from opt import opt
from tqdm import tqdm
from pathlib import Path
import time
from random import randrange

from vars import *
from detections import *

from tiah.tools import read_video, get_framesize
from tiah.vars import *
from tiah.get_data import get_bbox_dict
from tiah.Drawing import draw_header
from tiah.pose import *

from tiah.Drawing import *
import matplotlib.pyplot as plt
# draw_header()
# from main import get_bbox_dict, filter_list, get_list, read_video
from main import *
import math
import time
from tqdm import tqdm


def extract_mags(video):
    cap, atts = read_video(video)
    name_desc = tqdm(range(atts[LENGTH]))
    count = 0

    fightModel = FightDetection()
    mag_mean_list, mag_sum_list = [], []
    prev_frame = None
    while 1:
        ret, frame = cap.read()

        if ret is False:
            break

        frame = cv2.resize(frame, (320, 240))

        "-------------------------MAIN MODULE----------------------"

        dense_frame, vis_frame, mag_sum = fightModel.dense(prev_frame, frame)
        mag_array = mag_sum.flatten()
        mag_array_sum = np.sum(mag_array)

        if math.isinf(mag_array_sum):
            b = np.logical_not(np.isinf(mag_array).astype(np.bool))
            mag_array = mag_array[b]
            mag_sum_list.append(np.sum(mag_array))
        else:
            mag_sum_list.append(mag_array_sum)

        cv2.imshow('11', cv2.resize(frame, (920, 720)))
        if cv2.waitKey(1) == ord('q'):
            break
        prev_frame = dense_frame
        name_desc.update(1)
        count += 1

    t = np.arange(len(mag_sum_list))

    outpath = os.path.join('/home/peter/dataset/data/mags', Path(video).stem + '.npy')
    res = np.array([t, mag_sum_list])
    np.save(outpath, res)
    cap.release()

def extract_bgs(video):
    ######################################
    # VGA resolution --> no morphoge
    ##################################
    from detections import PaperDetection
    outpath = os.path.join('/home/peter/workspace/dataset/gist/elevator/fgmask', Path(video).stem + '.npy')

    if os.path.exists(outpath):
        print("passing", outpath)
        return

    yolotxt_path = '/home/peter/workspace/dataset/gist/elevator/yolotxt'

    cap, atts = read_video(video)
    bbox_txt_path = os.path.join(yolotxt_path, f'yolo_{Path(video).stem}.txt')
    bbox_list_dict = get_bbox_dict(bbox_txt_path)
    name_desc = tqdm(range(atts[LENGTH]))
    count = 0
    paperModel = PaperDetection()

    fgmask_list = []
    cnt_area_list_list = []
    while 1:
        ret, frame = cap.read()
        if ret is False:
            break
        fg_frame = cv2.resize(frame, (320, 240))

        bbox_list = get_list(bbox_list_dict, count)
        person_list = filter_list(bbox_list)
        # fgmask = paperModel.apply(frame, count, person_list)
        fgmask = paperModel.apply(fg_frame, count, resize_bbox_list(person_list, frame.shape,(320,240)))
        vis_frame, paper_flag, paper_count,cnt_area_list = paperModel.detect(fgmask)

        fgmask_list.append(fgmask)
        cnt_area_list_list.append(cnt_area_list)


        name_desc.update(1)
        count += 1
    # outpath = os.path.join('/home/peter/dataset/data/fgmask', Path(video).stem + '.npy')
    res = np.array([ fgmask_list ,cnt_area_list_list])
    np.save(outpath, res)
    cap.release()
    name_desc.close()
