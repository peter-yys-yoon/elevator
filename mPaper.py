import os, sys

PROJECT_PATH = '/home/peter/workspace/code/elev'
sys.path.append(PROJECT_PATH)

import cv2
import numpy as np
import pickle
import os


class BGsubs:
    def __init__(self):
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        # self.fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        # self.fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
        self.fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
        self.bg_thres = 100


    def apply(self, frame, count, bbox_list):
        if count > self.bg_thres:
            fgmask = self.fgbg.apply(frame, learningRate=0)
        else:
            fgmask = self.fgbg.apply(frame, learningRate=-1)
        th1,th2 = 0.9, 1.1

        for bbox in bbox_list:
            idx, x1, y1, x2, y2, conf, cls_conf, cls_pred = bbox
            x1,y1,x2,y2 = int(x1*th1),int(y1*th1),int(x2*th2),int(y2*th2)
            fgmask[y1:y2, x1:x2] = 0

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        # ret, thresh = cv2.threshold(fgmask, 129, 255, cv2.THRESH_TRUNC)
        thresh =fgmask
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        fgmask = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)


        for cnt in contours:
            if cv2.contourArea(cnt) > 500:
                cv2.drawContours(fgmask, [cnt], -1, (0, 255, 0), 3)

        resized_mask = cv2.resize(fgmask, (640, 480))
        resized_org = cv2.resize(frame, (640, 480))
        # draw_header(resized_org,count, COLOR_BLACK)
        # print('Image:' ,count , ' Update:' , update)

        return np.hstack((resized_mask, resized_org))

