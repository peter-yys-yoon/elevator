import os,sys

PROJECT_PATH = '/home/peter/workspace/code'
sys.path.append(PROJECT_PATH)

import cv2
import numpy as np
from tiah.Drawing import draw_header
from tiah.vars import COLOR_BLACK


def bs(videopath):
    print('video:', videopath)
    cap = cv2.VideoCapture()
    cap.open(videopath)
    print(cap.isOpened())
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
    fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
    # fgbg = cv2.createBackgroundSubtractorMOG2()
    # fgbg.setDetectShadows(True)

    count = 0
    while 1:
        ret, frame = cap.read()
        if ret is False:
            break

        if count > 100:
            update = 0
        else:
            update = -1

        fgmask = fgbg.apply(image=frame, learningRate = update)
        # fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        # print(np.unique(fgmask.reshape(-1)))

        ret, thresh = cv2.threshold(fgmask, 129, 255, cv2.THRESH_TRUNC)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # print(np.unique(thresh.reshape(-1)))
        # print('--------------------------------')
        fgmask = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

        for cnt in contours:
            if cv2.contourArea(cnt) > 200:
                cv2.drawContours(fgmask, [cnt], -1, (0, 255, 0), 3)



        resized_mask = cv2.resize(fgmask, (640,480))
        resized_org = cv2.resize(frame, (640,480))
        draw_header(resized_org,count, COLOR_BLACK)
        # print('Image:' ,count , ' Update:' , update)
        cv2.imshow('frame', np.hstack((resized_mask, resized_org)))
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break
        count += 1



videopath = '/home/peter/dataset/gist_elevator/raw_videos/C001A004D001P0003T0002.avi'

bs(videopath)

