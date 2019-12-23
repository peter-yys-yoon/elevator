
from __future__ import division

import os
import sys
sys.path.append('/home/peter/workspace/code')
sys.path.append('/home/peter/workspace/code/PyTorch-YOLOv3')
sys.path.append('/home/peter/workspace/code/PyTorch-YOLOv3/utils')
# print(sys.path)


from yolo.models import *
# from yolo.models import *
from yolo.utils.utils import *
from yolo.utils.datasets import *

from opt import opt
from vars import *
from tiah.tools import int2round, get_properties
from tiah.vars import *
from tiah.ntutool import *

import time
import argparse
from tqdm import tqdm


import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch
import torch.nn.functional as F
from yolo.utils.augmentations import horisontal_flip
from torch.utils.data import Dataset
from yolo.utils.datasets import resize,pad_to_square
import torchvision.transforms as transforms

import cv2


def detect(img):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # opt = parser.parse_args()
    # print(opt)
    args = opt
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("output", exist_ok=True)

    # Set up model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    model.eval()  # Set in evaluation mode

    dataloader = DataLoader(
        ImageFolder(opt.image_folder, img_size=opt.img_size),
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
    )

    classes = load_classes(opt.class_path)  # Extracts class labels from file
    # print('cuda available? ', torch.cuda.is_available())
    # quit()
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index

    print("\nPerforming object detection:")
    prev_time = time.time()

    # 전단지
    # videopath = '/home/peter/dataset/gist_elevator/raw_videos/C001A004D001P0003T0002.avi'

    #기절
    # videopath = '/home/peter/dataset/gist_elevator/raw_videosC001A002D001P0003T0005.avi'

    # videopath = '/home/peter/extra/dataset/gist/elevator/C001A002D001P0009T0001.avi'
    videopath = '/home/peter/workspace/dataset/gist/elevator/C001A004D001P0003T0002.avi'
    # videopath = args.video
    
    
    videoname = os.path.basename(videopath)
    print(videoname)
    count = 0
    WAIT = 1
    
    cap = cv2.VideoCapture()
    cap.open(videopath)
    assert cap.isOpened(), 'Video is not opened'
    pps = get_properties(cap) # fps, fourcc, w, h , length

    w,h = pps[WIDTH], pps[HEIGHT]
    
    framesize = int2round( (w/2, h/2)) # ORG
    # framesize = int2round( (w/2, h/2)) # HALF
    # framesize = int2round( (h/2,w/2)) # ROTATE

    if args.save_video:
        # savepath = f'/home/peter/workspace/code/elev/output/rotate_yolo_{videoname}'
        #OBAMA
        savepath = f'/home/peter/extra/Workspace/code/elev/output/yolo/yolo_{videoname}'
        #ROTATE
        # savepath = f'/home/peter/extra/Workspace/code/elev/output/rotate_yolo_{videoname}'

        savepath = f'{PROJECT_PATH}/output/yolo/yolo_{videoname}'
        print(savepath)
        if os.path.exists(savepath):
            quit()

        writer = cv2.VideoWriter(savepath, XVID, 25, framesize)



    name_desc = tqdm(range(pps[LENGTH]))
    while 1:
        ret, frame = cap.read()
        if ret is False:
            break

        out = frame.copy()
        
        # out = cv2.transpose(frame)
        # out = cv2.flip(out, flipCode=1)
        vis_frame = out.copy()

        img = transforms.ToTensor()(out)  # (3, W, H)
        img, _ = pad_to_square(img, 0)  # convert to N x N squre matrix, (3, W, W)
        img = resize(img, opt.img_size)  # (3, M, M)
        
        img = img.unsqueeze(0)
        input_imgs = Variable(img)




        
        with torch.no_grad():
            detections = model(input_imgs.cuda())
            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)
        

        if detections[0] is not None:
            # Rescale boxes to original image
            detections = rescale_boxes(detections[0], opt.img_size, vis_frame.shape[:2])
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                cv2.rectangle(vis_frame, (x1,y1), (x2,y2), (255,33,133),2)
                        
        vis_frame = cv2.resize(vis_frame, framesize)

        if args.save_video:
            writer.write(vis_frame)
        if args.vis:
            cv2.imshow('11', vis_frame)
            if cv2.waitKey(WAIT) == ord('q'):
                break

        name_desc.update(1)

    if args.save_video:
        writer.release()
        
