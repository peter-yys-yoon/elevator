import os, sys
# from skimage.feature import hog

PROJECT_PATH = '/home/peter/workspace/code/elev'
sys.path.append(PROJECT_PATH)

import cv2
import numpy as np
import pickle
import os

import pdb
class PaperDetection:
    def __init__(self, bg_thres=100, THRES_CNT_AREA=400 ):
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        # self.fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        # self.fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
        self.fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
        self.bg_thres = 100
        self.THRES_CNT_AREA = 400
        self.paper_flag_count = 0

    def apply(self, frame, count, bbox_list):
        if count > self.bg_thres:
            fgmask = self.fgbg.apply(frame, learningRate=0)
        else:
            fgmask = self.fgbg.apply(frame, learningRate=-1)
        th1,th2 = 0.9, 1.1 #TODO check here



        invalid_width = int(frame.shape[1]*0.3)
        fgmask[:,0:invalid_width] = 0

        no_person_mask = np.array(fgmask)

        for bbox in bbox_list:
            # idx, x1, y1, x2, y2, conf, cls_conf, cls_pred = bbox
            x1, y1, x2, y2  = bbox
            x1,y1,x2,y2 = int(x1*th1),int(y1*th1),int(x2*th2),int(y2*th2)
            no_person_mask[y1:y2, x1:x2] = 0

        # resized_mask = cv2.resize(fgmask, (640, 480))
        # resized_org = cv2.resize(frame, (640, 480))
        # draw_header(resized_org,count, COLOR_BLACK)
        # print('Image:' ,count , ' Update:' , update)

        # return np.hstack((resized_mask, resized_org))
        return no_person_mask, fgmask

    def detect(self, _fgmask, vis_frame):
        # CHECK!!! morpohogy!!!!!!!!!!
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        # _fgmask = cv2.morphologyEx(_fgmask, cv2.MORPH_OPEN, kernel)


        # ret, thresh = cv2.threshold(_fgmask, 129, 255, cv2.THRESH_TRUNC)
        thresh = _fgmask
        # _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        fgmask = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        font = cv2.FONT_HERSHEY_COMPLEX

        xr, xy  = (1920/320) , (1080/240)

        paper_count = 0
        for cnt in contours:
            if cv2.contourArea(cnt) > 40:
                # pdb.set_trace()
                cnt[:,0,0] = cnt[:,0,0] * xr
                cnt[:,0,1] = cnt[:,0,1] * xy
                cv2.drawContours(vis_frame, [cnt], -1, (0, 0, 255), -1)
                paper_count += 1
                approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
                cv2.drawContours(fgmask, [approx], 0, (0), 5)



        cnt_area_list = []
        for cnt in contours:
            cnt_area_list.append(cv2.contourArea(cnt))

        cv2.putText(fgmask, f'paper{paper_count}', (50, 150), font, 1, (0, 255, 0))
        if paper_count:
            self.paper_flag_count += 1

        if self.paper_flag_count > 60:
            return fgmask , True ,self.paper_flag_count, cnt_area_list,vis_frame
        else:
            return fgmask, False ,self.paper_flag_count,cnt_area_list,vis_frame


class FightDetection:
    def __init__(self):

        self.hog = cv2.HOGDescriptor()

    def dense(self,prev_frame, frame):
        next = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_frame is not None:
            hsv = np.zeros_like(frame)
            hsv[..., 1] = 255
            flow = cv2.calcOpticalFlowFarneback(prev_frame, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            # print(mag.shape)
            return next ,rgb , mag
        else:
            return next, frame , np.array([0])







    def histogram(self, frame, bbox_list):

        for bbox in bbox_list:
            idx, x1, y1, x2, y2, conf, cls_conf, cls_pred = bbox
            person_area = frame[y1:y2, x1:x2 , :]
            fd = self.hog.compute(person_area)


            #
            # fd, hog_image = hog(person_area, orientations=8, pixels_per_cell=(16, 16),
            #                 cells_per_block=(1, 1), visualize=True, multichannel=True)
            #
            #
            print(person_area.shape, fd.shape)



class FallDetection:
    def __init__(self, threshold):
        self.counter = 0
        self.fall_threshold = threshold

    def detect(self, bbox_list, curr_mag_sum ,FIGHT_FLAG_MAGS_THRESHOLD):

        for bbox in bbox_list:
            idx, x1, y1, x2, y2, conf, cls_conf, cls_pred = bbox

            if self.aspec_ratio([x1, y1, x2, y2]):
                # if curr_mag_sum < FIGHT_FLAG_MAGS_THRESHOLD:
                self.counter += 1
                break

        if self.counter > self.fall_threshold:
            return self.counter, True
        else:
            return self.counter, False

    def aspec_ratio(self, bbox):
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1

        if w / h > 1:  # width is larger,
            return True
        else:  # height is larger
            return False




