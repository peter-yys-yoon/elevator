"------------------------PYTHON-------------------------------"
import os
import cv2
import math
from opt import opt
from tqdm import tqdm
from pathlib import Path
import time
from random import randrange
import platform
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

# FIGHT1=''
# FIGHT2
#
# 'A001D001', 'A001D002', 'A002D001', 'A002D002', 'A003D001', 'A004D001', 'A005D001', 'A006D001', 'A006D002',
#           'A006D003'

"""--------------------------------------------"
A001: fight --> 0
A002: Fall --> 1
A003: Panic --> 2
A004: Paper --> 3
A005: Escape --> 4
A006: Standing --> 5
"--------------------------------------------"""
label_long = ['A001D001', 'A001D002', 'A002D001',
              'A002D002', 'A003D001', 'A004D001',
              'A005D001', 'A006D001', 'A006D002', 'A006D003']

label_short = ['A001', 'A002', 'A003', 'A004', 'A005', 'A006']

exclude_list = ['C001A003D001P0007T0001', 'C001A006D001P0006T0001', 'C001A003D001P0003T0001']

A001 = 'A001'  # 134 = 79 + 55
A002 = 'A002'  # 62 = 56 + 6
A003 = 'A003'  # 52
A004 = 'A004'  # 46
A005 = 'A005'  # 48
A006 = 'A006'  # 133 = 52 + 58 + 23
A006D001 = 'A006D001'  # 52
A006D002 = 'A006D002'  # 52
A006D003 = 'A006D003'  # 52

preset_mm = randrange(1, 46)
cam_id = randrange(1, 6)


def get_dy_list():
    if platform.node() == 'obama':
        path = f'./data/dyselect/good'

    else:
        path = f'/home/peter/dataset/data/dyselect/good'

    flist = os.listdir(path)
    flist.sort()
    fflist = []
    for f in flist:
        ff = f.split('_')[1]
        fflist.append(os.path.join(PATH_DATA_VIDEO, ff))
    return fflist


def get_time_string(count):
    time_in_sec = count // 30
    mm, ss = divmod(time_in_sec, 60)
    hh = 17

    return f'{hh}:{mm + preset_mm}:{str(ss).zfill(2)}'


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


def print_log_sorting(msg):
    with open(os.path.join('/home/peter/workspace/projects/xiah/Elev', 'log_sorting.txt'), 'a') as f:
        print(msg, file=f)


def print_log(msg, logname, printing=True, print_time=False):
    if printing:
        print(msg)
    if print_time:
        localtime = time.asctime(time.localtime(time.time()))
        msg = "[ " + localtime + ' ] ' + msg
    with open(os.path.join('/home/peter/workspace/projects/xiah/Elev', logname), 'a') as f:
        print(msg, file=f)


def resize_bbox_list(bbox_list, before_shape, after_shape):
    if len(bbox_list):

        beore_w, before_h = before_shape[1], before_shape[0]  # np.shape
        after_w, after_h = after_shape[0], after_shape[1]  # cv2.shape
        new_bbox_list = np.array(bbox_list)  # x1 x2 y1 y2
        new_bbox_list = new_bbox_list[:, 1:5]
        new_bbox_list = new_bbox_list.astype(np.float64)
        # idx, x1, y1, x2, y2, conf, cls_conf, cls_pred = bbox

        # print(new_bbox_list.shape , new_bbox_list.dtype, new_bbox_list)
        # quit()
        new_bbox_list[:, 0] *= after_w / beore_w
        new_bbox_list[:, 2] *= after_w / beore_w
        new_bbox_list[:, 1] *= after_h / before_h
        new_bbox_list[:, 3] *= after_h / before_h

        return new_bbox_list.astype(np.uint32)
    else:
        return []


def get_label(ff):
    # tag = ff[4:12]
    tag = ff[4:8]
    # aa = label_long
    aa = label_short
    idx = aa.index(tag) + 1
    if idx == 5:
        idx = 3
    return idx


def filter_video_list(_video_list):
    video_list = []
    f = open(os.path.join('/home/peter/workspace/projects/xiah/Elev', 'exclude.txt'), 'r')
    lines = [line.rstrip('\n') for line in f]
    for _v in _video_list:
        if _v not in lines:
            video_list.append(_v)

    return video_list


def make_video_filter_by_count(detection_start_threshold=60):
    # filter videos which includes more than one person where video belongs to single-person action.

    if platform.node() == 'obama':
        BASEPATH = '/home/peter/workspace/dataset/gist/elevator'
    else:
        BASEPATH = '/home/peter/dataset/data'

    yolotxt_path = os.path.join(BASEPATH, 'yolotxt')
    video_path = os.path.join(BASEPATH, 'video')

    video_list = os.listdir(video_path)

    for video in video_list:
        tmpa = video.index('A')
        vv = video[tmpa:tmpa + 4]

        continue_flag = True
        for action in [A002, A003, A004, A005, A006D001]:  # less than two person
            if action in video:
                continue_flag = False
                break
        if continue_flag:
            continue

        bbox_txt_path = os.path.join(yolotxt_path, f'yolo_{Path(video).stem}.txt')
        bbox_list_dict = get_bbox_dict(bbox_txt_path)

        person_count_list = []
        cap, atts = read_video(os.path.join(video_path, video))
        total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        person_first_enter_counter = 0

        for count in range(int(total_frame_count)):

            bbox_list = get_list(bbox_list_dict, count)
            person_list = filter_list(bbox_list)
            person_count = len(person_list)

            curr_person_count = len(person_list)
            if curr_person_count:
                person_first_enter_counter += 1

            if person_first_enter_counter < detection_start_threshold:
                continue

            person_count_list.append(person_count)

            if np.mean(person_count_list) > 1:
                with open(os.path.join('/home/peter/workspace/projects/xiah/Elev', 'exclude.txt'), 'a') as f:
                    print(Path(video).stem, file=f)
                break


###############################################################################################

def bbbbbbbbbbbbbbbbbb(video, a=30, b=50, c=60000, d=30, e=60000, g=60, debug=False):
    args = opt
    detection_start_threshold = a
    FIGHT_FLAG_COUNTER_THRESHOLD = b
    FIGHT_FLAG_MAGS_THRESHOLD = c
    PANIC_FLAG_COUNTER_THRESHOLD = d
    PANIC_FLAG_MAGS_THRESHOLD = e
    panic_flag_counter = 0
    fight_flag_counter = 0

    if platform.node() == 'obama':
        BASEPATH = '/home/peter/workspace/dataset/gist/elevator'
    else:
        BASEPATH = '/home/peter/dataset/data'

    yolotxt_path = os.path.join(BASEPATH, 'yolotxt')
    npypath = os.path.join(BASEPATH, 'mags')
    fgmaskpath = os.path.join(BASEPATH, 'fgmask')
    video_path = os.path.join(BASEPATH, 'video')

    bbox_txt_path = os.path.join(yolotxt_path, f'yolo_{Path(video).stem}.txt')
    bbox_list_dict = get_bbox_dict(bbox_txt_path)
    dense_npy = np.load(os.path.join(npypath, f'{Path(video).stem}.npy'))[1]
    fgmask_cnt_npy = np.load(os.path.join(fgmaskpath, f'{Path(video).stem}.npy'), allow_pickle=True)
    fgmask_list = fgmask_cnt_npy[0]
    cnt_areA_list = fgmask_cnt_npy[1]
    # print(fgmask_list.shape)
    # quit()

    person_count_list = []
    cap, atts = read_video(os.path.join(video_path, video))
    total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    person_first_enter_counter = 0

    paperModel = PaperDetection()
    fightModel = FightDetection()
    fall_Model = FallDetection(threshold=g)
    predict = -1
    # for count in range(int(total_frame_count * 0.8)):
    for count in range(int(total_frame_count)):


        bbox_list = get_list(bbox_list_dict, count)
        person_list = filter_list(bbox_list)
        person_count = len(person_list)

        if args.vis:
            _, vis_frame = cap.read()
            for bbox in person_list:  # drawing bbox
                idx, x1, y1, x2, y2, conf, cls_conf, cls_pred = bbox
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), COLOR_GREEN, 1)

        curr_person_count = len(person_list)
        if curr_person_count:
            person_first_enter_counter += 1

        if person_first_enter_counter < detection_start_threshold:
            continue

        person_count_list.append(person_count)

        curr_mags_sum = dense_npy[count]
        if curr_mags_sum > FIGHT_FLAG_MAGS_THRESHOLD:
            fight_flag_counter += 1
        if curr_mags_sum > PANIC_FLAG_MAGS_THRESHOLD:
            panic_flag_counter += 1

        curr_fgmask = fgmask_list[count]
        _, paper_flag, paper_counter, _ ,_= paperModel.detect(curr_fgmask, np.ones((30,30)))

        fall_counter, fall_flag = fall_Model.detect(person_list, curr_mags_sum, PANIC_FLAG_MAGS_THRESHOLD)

        if np.mean(person_count_list) > 1:
            if fight_flag_counter > FIGHT_FLAG_COUNTER_THRESHOLD:
                predict = 1  # fight

        else:
            if panic_flag_counter > PANIC_FLAG_COUNTER_THRESHOLD:
                predict = 3  # Panic
            elif fall_flag:
                predict = 2  # Fall
            elif paper_flag:
                predict = 4  # Paper

        if debug:
            # print(f'Video({Path(video).stem}), Frame({count}), #person({curr_person_count}), #avgperson({np.mean(person_count_list):.2f}), FallCount({fall_counter}), {panic_flag_counter} > {PANIC_FLAG_COUNTER_THRESHOLD}?, ->Pred({predict})')
            print(
                f'Video({Path(video).stem}), Frame({count}), Paper({paper_counter, paper_flag}), FallCount({fall_counter}), Panic({panic_flag_counter}), ->Pred({predict})')
            time.sleep(0.001)

        if predict == -1:
            predict = 6
        if predict < 6:
            break

        if args.vis:
            cv2.imshow('11', cv2.resize(vis_frame, (920, 720)))
            if cv2.waitKey(1) == ord('q'):
                break
    if debug:
        input('<<<<<<<<<<')
    cap.release()

    return predict, get_label(Path(video).stem)


def optimize_panic():
    max_acc, max_param, maxx_acc_list = 0, 0, []
    log_name = 'log_panic.txt'
    a_range = range(60, 100, 30)  # detection_delay
    d_range = range(50, 80, 10)  # panic_count
    e_ragne = range(18000, 20000, 10000)  # panic_mags
    g_range = range(30, 100, 10)  # fall_count

    count = 0
    for d in d_range:  # PANIC_FLAG_COUNTER_THRESHOLD:
        for a in a_range:  # detection_start_threshold
            for e in e_ragne:  # PANIC_FLAG_COUNTER_THRESHOLD
                for g in g_range:

                    count += 1
                    acc, acc_list, count_list = aaaaaaaaaaaaaaaa(a, 40, 40000, d, e, g)
                    print(f'{count}/{len(a_range) * len(d_range) * len(e_ragne) * len(g_range)} Params:{(a, e, d, g)} ')

                    if acc > max_acc:
                        max_acc = acc
                        max_acc_list = acc_list
                        max_param = (a, e, d)
                        print_log('---------------------------------------------', log_name, printing=True)
                        print_log(f'Panic Params:{max_param}', log_name, printing=True)
                        print_log(f'Accuracy: {acc} %', log_name, printing=True)
                        print_log(' '.join([f'A00{i + 1}({xx:.1f}%)' for i, xx in enumerate(max_acc_list)]), log_name,
                                  printing=True)

    print_log('Final Result <<<<<<<<<<<<<<<<<<', log_name)
    print_log(f'Params:{max_param}', log_name)
    print_log(f'Accuracy: {max_acc} %', log_name)
    print_log(' '.join([f'A00{i + 1}({xx:.1f}%)' for i, xx in enumerate(maxx_acc_list)]), log_name)

def optimize_fight():
    max_acc, max_param, max_acc_list = 0, 0, []
    log_name = 'log_fight.txt'
    for a in [45, 60, 90]:  # detection_start_threshold
        for c in range(20000, 30000, 5000):  # FIGHT_FLAG_MAGS_THRESHOLD
            for b in range(165, 200, 15):  # FIGHT_FLAG_COUNTER_THRESHOLD:
                acc, acc_list, count_list = aaaaaaaaaaaaaaaa(a, b, c, 30, 40000, 60)

                if acc > max_acc:
                    max_acc = acc
                    max_acc_list = acc_list
                    max_param = (a, b, c)
                    print_log('---------------------------------------------', log_name, printing=True)
                    print_log(f'Params:{max_param}', log_name, printing=True)
                    print_log(f'Accuracy: {acc} %', log_name, printing=True)
                    print_log(' '.join([f'A00{i + 1}({xx:.1f}%)' for i, xx in enumerate(acc_list)]), log_name,
                              printing=True)

    print_log('Final Result <<<<<<<<<<<<<<<<<<', log_name)
    print_log(f'Params:{max_param}', log_name)
    print_log(f'Accuracy: {max_acc} %', log_name)
    print_log(' '.join([f'A00{i + 1}({xx:.1f}%)' for i, xx in enumerate(max_acc_list)]), log_name)


def aaaaaaaaaaaaaaaa(a, b, c, d, e, g, debug=False):
    if platform.node() == 'obama':
        video_path = '/home/peter/workspace/dataset/gist/elevator/video'
    else:
        video_path = '/home/peter/dataset/data/video'
    video_list = os.listdir(video_path)
    video_list = filter_video_list(video_list)

    name_desc = tqdm(range(len(video_list)))
    label_list, pred_list = [], []

    for video in video_list:
        name_desc.update(1)

        continue_flag = True
        # for action in [A002, A003, A004, A005, A006D001]:  # less than two person
        for action in [A004]:  # less than two person
            # for action in [A001,  A006D002,A006D003]: # more than two person
            if action in video:
                continue_flag = False
                break
        # if continue_flag:
        #     continue

        pred, label = bbbbbbbbbbbbbbbbbb(os.path.join(video_path, video), a, b, c, d, e, g, False)
        pred_list.append(pred)
        label_list.append(label)

    acc = round(100 * np.mean(np.equal(pred_list, label_list).astype(np.int32)), 1)
    label_list = np.array(label_list)
    pred_list = np.array(pred_list)

    action_acc_list, action_count_list = [], []
    for i, action in enumerate(label_short):
        mask = np.where(label_list == i + 1)
        if mask[0].shape[0]:
            aa = round(100 * np.mean(np.equal(pred_list[mask], label_list[mask]).astype(np.int32)), 1)
            bb = f'{np.sum(np.equal(pred_list[mask], label_list[mask]).astype(np.int32))}/{mask[0].shape[0]}'
        else:
            aa, bb = 0, '0/0'

        action_acc_list.append(aa)
        action_count_list.append(bb)
    name_desc.close()
    return acc, action_acc_list, action_count_list




if __name__ == "__main__":

    arg = opt
    if arg.tmp == 'panic':
        optimize_panic()
    elif arg.tmp == 'fight':
        optimize_fight()
    else:
        # make_video_filter_by_count(60)

        acc, acc_list, count_list = aaaaaaaaaaaaaaaa(60, 165, 20000, 60, 18000, 60, False)
        print('---------------------------------------------')
        print(f'Accuracy: {acc} %')
        print(' '.join([f'A00{i + 1}({xx:.1f}%)' for i, xx in enumerate(acc_list)]))
        print(' '.join([f'A00{i + 1}({xx})' for i, xx in enumerate(count_list)]))

    # video_reading()

    # for video in os.listdir(PATH_DATA_VIDEO):
    #     if 'A001D001' in video:
    #         opt.video = os.path.join(PATH_DATA_VIDEO, video)  # fight, kym
    #         video_reading()
    #         print(video)
    #         quit()

    "----------------------------------"
