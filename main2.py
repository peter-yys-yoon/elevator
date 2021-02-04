"------------------------PYTHON-------------------------------"
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

"""--------------------------------------------"
A001: fight --> 1
A002: Fall --> 2
A003: Panic --> 3
A004: Paper --> 4
A005: Escape --> 5
A006: Standing --> 6
"--------------------------------------------"""
label_long = ['A001D001', 'A001D002', 'A002D001', 'A002D002', 'A003D001', 'A004D001', 'A005D001', 'A006D001', 'A006D002',
          'A006D003']
label_short= ['A001', 'A002', 'A003', 'A004', 'A005', 'A006']
preset_mm = randrange(1, 46)
cam_id = randrange(1, 6)







def mask_counter():
    # npypath ='/home/peter/workspace/dataset/gist/elevator/mags'
    npypath = '/home/peter/dataset/data/mags'

    flist = os.listdir(npypath)
    plt.cla()

    thres = 25
    # actions = ['A001D001', 'A001D002', 'A002', 'A003', 'A004', 'A005', 'A006D001', 'A006D002', 'A006D003']
    actions = [ 'A003', 'A005', 'A006D001']
    # actions = [ 'A003', 'A005', 'A006D001']
    # actions = ['A001', 'A002', 'A003', 'A004', 'A005', 'A006']

    A001 = 'A001'  # 134 = 79 + 55
    A002 = 'A002'  # 62
    A003 = 'A003'  # 52
    A004 = 'A004'  # 46
    A005 = 'A005'  # 48
    A006 = 'A006'  # 133 = 52 + 58 + 23
    A006D001 = 'A006D001'  # 52

    # plt.subplot(3, 3, plot_idx)
    max_count= 0
    max_thres =0

    for mag_thres in range(10000,20000,700):
        for count_thres in range(30,150,8):
            fcount_list,mcount_list = [],[]
            for aidx, action in enumerate([A003]):
            # for aidx, action in enumerate([A003, A005, A006D001]):
                fcount, mcount = 0,0
                for f in flist:
                    if action in f:
                        fcount += 1
                        dense_npy = np.load(os.path.join(npypath, f))
                        mask = np.where(dense_npy[1] > mag_thres)
                        over_count = mask[0].shape[0]
                        if over_count > count_thres:
                            mcount +=1
                            # print(f, count_thres)
                            # time.sleep(0.05)

                fcount_list.append(fcount)
                mcount_list.append(mcount)

            total_count = np.sum(mcount_list)/np.sum(fcount_list)
            if 0.8 < total_count:
                # max_count = total_count

                # print(thres, np.sum(mcount_list), np.sum(fcount_list))
                tmplist= []
                for i in range(len(fcount_list)):
                    tmplist.append(round(100* mcount_list[i]/fcount_list[i],1))
                print(mag_thres, count_thres, '--->','% '.join(str(xx) for xx in tmplist))



    # print(action, mcount, fcount, round(mcount/fcount,1), end=' // ')
    # print()



def plot_dense():
    # npypath ='/home/peter/workspace/dataset/gist/elevator/mags'
    npypath = '/home/peter/dataset/data/mags'

    flist = os.listdir(npypath)
    plt.cla()
    # plt.title(f)
    # plt.ylim([0,300000])
    # thres =100,000
    thres = 25
    # actions = ['A001D001', 'A001D002', 'A002', 'A003', 'A004', 'A005', 'A006D001', 'A006D002', 'A006D003']
    actions = [ 'A003', 'A005', 'A006D001']
    # actions = ['A001', 'A002', 'A003', 'A004', 'A005', 'A006']
    # markers = ['r-', 'b-', 'g-','y-','m-','c-']
    markers = ['ro', 'bo', 'go', 'yo', 'mo', 'co', 'ro', 'bo', 'go', ]
    plot_idx = 0
    A001 = 'A001'  # 134 = 79 + 55
    A002 = 'A002'  # 62
    A003 = 'A003'  # 52
    A004 = 'A004'  # 46
    A005 = 'A005'  # 48
    A006 = 'A006'  # 133 = 52 + 58 + 23
    A006D001 = 'A006D001'  # 52

    # plt.subplot(3, 3, plot_idx)


    a003,a005,a006 =[],[],[]
    for f in flist:
        # if not f =='C001A001D001P0011T0005.npy':
        #     continue

        tmpa = f.index('A')
        vv = f[tmpa:tmpa + 4]
        plt.ylim([0, 100000])
        continue_flag = True
        # for action in [A003,A004,A006D001]:
        for action in [A003, A005, A006D001]:
            # print(f'{action} in {video} {action in video}')
            if action in f:
                continue_flag = False
                break

        if continue_flag:
            continue

        dense_npy = np.load(os.path.join(npypath, f))
        mask = np.where(dense_npy[1] > thres)
        if vv =='A003':
            # plt.plot(dense_npy[0][mask], dense_npy[1][mask], 'bo')
            a003.append(mask[0].shape[0])
        elif vv =='A005':
            # plt.plot(dense_npy[0][mask], dense_npy[1][mask], 'go')
            a005.append(mask[0].shape[0])
        elif vv =='A006':
            # plt.plot(dense_npy[0][mask], dense_npy[1][mask], 'ro')
            a006.append(mask[0].shape[0])
        # plt.show()
    print(np.mean(a003), np.std(a003))
    print(np.mean(a005),np.std(a005))
    print(np.mean(a006),np.std(a006))

                # plt.plot(dense_npy[0], dense_npy[1], marker)






if __name__ == "__main__":
    pass
    # mask_counter()
