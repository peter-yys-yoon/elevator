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
from main import *


# draw_header()


def video_reading():
    args = opt
    WAIT = 1
    print(args.video)
    cap, atts = read_video(args.video)
    framesize = get_framesize(cap)

    bbox_txt_path = os.path.join(PATH_DATA_YOLOTXT, f'yolo_{Path(args.video).stem}.txt')
    bbox_list_dict = get_bbox_dict(bbox_txt_path)
    # pose_json_path = os.path.join(PATH_DATA_JSONJSON, f'alpha_{Path(args.video).stem}.json')
    # pose_list_dict = get_pose_list_from_json(pose_json_path)

    detection_start_threshold = 60
    FIGHT_FLAG_COUNTER_THRESHOLD = 165
    FIGHT_FLAG_MAGS_THRESHOLD = 20000
    PANIC_FLAG_COUNTER_THRESHOLD = 60
    PANIC_FLAG_MAGS_THRESHOLD = 18000
    FALL_FLAG_COUNTER_TREHSHOLD = 60
    panic_flag_counter = 0
    fight_flag_counter = 0
    person_first_enter_counter = 0

    if args.save_video:
        savepath = f'{args.outdir}/{Path(args.video).stem}.avi'
        print('Video writing at ', savepath)
        writer = cv2.VideoWriter(savepath, CODEC_XVID, 30, (920, 720))

    name_desc = tqdm(range(atts[LENGTH]))
    count = 0

    fallmodel = FallDetection(FALL_FLAG_COUNTER_TREHSHOLD)
    paperModel = PaperDetection()
    fightModel = FightDetection()
    predict = 7

    prev_frame = None
    color = COLOR_BLACK
    action = ''
    ckpt = time.time()
    person_count_list = []

    while 1:
        ret, frame = cap.read()

        if ret is False:
            break

        name_desc.update(1)
        vis_frame = frame.copy()
        paper_vis_frame = vis_frame
        msg_frame = cv2.resize(frame, (320, 240))
        fg_frame = cv2.resize(frame, (320, 240))
        bbox_list = get_list(bbox_list_dict, count)
        person_list = filter_list(bbox_list)
        # pose_list= get_list(pose_list_dict,count)



        # name_desc.update(1)
        count += 1
        curr_person_count = len(person_list)
        person_count_list.append(curr_person_count)
        if curr_person_count:
            person_first_enter_counter += 1

        "-------------------------MAIN MODULE----------------------"

        ###############################
        # Background-substraction
        ressized_box_list = resize_bbox_list(person_list, vis_frame.shape, (320, 240))
        fgmask, org_fgmask = paperModel.apply(fg_frame, count, ressized_box_list)

        ###############################
        # Dense optical flow
        dense_frame, mag_vis_frame, mag_sum = fightModel.dense(prev_frame, msg_frame)
        prev_frame = dense_frame

        res_show = mag_vis_frame

        if person_first_enter_counter > detection_start_threshold:
            mag_array = mag_sum.flatten()
            mag_array_sum = np.sum(mag_array)
            if math.isinf(mag_array_sum):
                b = np.logical_not(np.isinf(mag_array).astype(np.bool))
                mag_array = mag_array[b]
                mag_array_sum = np.sum(mag_array)

            if mag_array_sum > FIGHT_FLAG_MAGS_THRESHOLD:
                fight_flag_counter += 1
            if mag_array_sum > PANIC_FLAG_MAGS_THRESHOLD:
                panic_flag_counter += 1

            # if detection_start_flag > THRES_DETECTION_START:
            fall_counter, fall_flag = fallmodel.detect(person_list, mag_array_sum, PANIC_FLAG_MAGS_THRESHOLD)
            _, paper_flag, paper_count, _ , paper_vis_frame= paperModel.detect(fgmask, np.array(vis_frame))
            # fall_flag = fallmodel.detect(bbox_list)
            # fight_flag = fightModel.histogram(frame,bbox_list)

            "-----------------------------------------------------------"
            # res += (40 * fgmask + gray) * 0.01
            res_show = mag_sum / mag_sum.max()
            res_show = np.floor(res_show * 255)
            res_show = res_show.astype(np.uint8)
            res_show = cv2.applyColorMap(res_show, cv2.COLORMAP_JET)
            # cv2.imshow('hitmap', res_show)

            if predict > 5:
                if np.mean(person_count_list) > 1:
                    if fight_flag_counter > FIGHT_FLAG_COUNTER_THRESHOLD:
                        predict = 1  # fight

                elif np.mean(person_count_list) > 0:
                    if panic_flag_counter > PANIC_FLAG_COUNTER_THRESHOLD:
                        predict = 3  # Panic
                    elif fall_flag:
                        predict = 2  # Fall
                    elif paper_flag:
                        predict = 4  # Paper

                else:
                    predict = 6

        if predict == 1:
            color = COLOR_RED
            action = 'Violence'
        elif predict == 2:
            color = COLOR_ORANGE
            action = 'Person-Fall'
            pass
        elif predict == 3:
            color = COLOR_CYAN
            action = 'Panic'
        elif predict == 4:
            color = COLOR_ORANGE
            action = 'Ad-attached'
        elif predict == 6:
            color = COLOR_GREEN
            action = 'Person Standing'
            # action = 'Ad-attached'
        else:
            color = COLOR_LIGHT_GREEN



        if args.save_video or args.save_video:
            header_label = get_header_label(count, person_list)
            aa = f'{header_label} {action} c{count}'
            draw_header(vis_frame, count, color, msg=aa, scale=1, thick=1)
            # draw_pose(vis_frame, pose_list)



            for bbox in person_list:  # drawing bbox
                idx, x1, y1, x2, y2, conf, cls_conf, cls_pred = bbox
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 1)

            resized_img = cv2.resize(vis_frame, (920, 720))
            resized_res = cv2.resize(res_show, (920, 720))
            resized_paper = cv2.resize(paper_vis_frame, (920, 720))

            final_vis_frame = cv2.addWeighted(resized_paper, 0.4, resized_img, 0.6, 0) # paper
            # final_vis_frame = cv2.addWeighted(resized_res, 0.2, resized_img, 0.8, 0) # fight

        if args.save_video:
            writer.write(final_vis_frame)
        if args.vis:
            cv2.imshow('11',final_vis_frame ) # paper


            if cv2.waitKey(WAIT) == ord('q'):
                break
    print(time.time() - ckpt, 'TTTTTTTTTAaaaaaaaaaaaaaaaaaaaaake')
    print(f'Average FPS:{atts[LENGTH]/(time.time() - ckpt)}')
    if args.save_video:
        writer.release()
    cap.release()
    name_desc.close()


if __name__ == "__main__":
    count = 0
    for video in os.listdir(PATH_DATA_VIDEO):

        if count > 8:
            break
        if 'A004' in video:
            count += 1
            opt.video = os.path.join(PATH_DATA_VIDEO, video)  # fight, kym
            video_reading()
            print(count, video)

