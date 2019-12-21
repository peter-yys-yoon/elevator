
from __future__ import division

import os
import sys
sys.path.append('/home/peter/workspace/code')
sys.path.append('/home/peter/workspace/code/PyTorch-YOLOv3')
sys.path.append('/home/peter/workspace/code/PyTorch-YOLOv3/utils')
# print(sys.path)


from yolov3.models import *
# from yolov3.models import *
from yolov3.utils.utils import *
from yolov3.utils.datasets import *

from opt import opt as args

from tiah_module.tools import int2round
import time
import datetime
import argparse
from tqdm import tqdm
from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch
import torch.nn.functional as F
from yolov3.utils.augmentations import horisontal_flip
from torch.utils.data import Dataset
from yolov3.utils.datasets import resize,pad_to_square
import torchvision.transforms as transforms

import cv2

def text_filled(frame, p1, label, color):
    txt_size, baseLine1 = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, FONT_SCALE, FONT_THINKESS)
    p1_ = (p1[0] - 10, p1[1] + 10)
    p2 = (p1[0] + txt_size[0] + 10, p1[1] - txt_size[1] - 10)
    cv2.rectangle(frame, p1_, p2, color, -1)
    cv2.putText(frame, label, p1, cv2.FONT_HERSHEY_DUPLEX, FONT_SCALE, WHITE, FONT_THINKESS)  # point is left-bottom


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    opt = parser.parse_args()
    print(opt)

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

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index

    print("\nPerforming object detection:")
    prev_time = time.time()


    # 전단지
    # videopath = '/home/peter/dataset/gist_elevator/raw_videos/C001A004D001P0003T0002.avi'

    #기절
    videopath = '/home/peter/dataset/gist_elevator/raw_videos/C001A002D001P0003T0005.avi'

    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    videoname = os.path.basename(videopath)



    cap = cv2.VideoCapture()
    cap.open(videopath)



    name_desc = tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frameSize = int2round( (h/2,w/2))

    savepath = f'/home/peter/workspace/code/elev/output/rotate_yolo_{videoname}'
    writer = cv2.VideoWriter(savepath, fourcc, 20, frameSize)

    print(videoname)
    print(savepath)
    count = 0
    while 1:
        name_desc.update(1)
        ret, frame = cap.read()
        if ret is False:
            break

        out = cv2.transpose(frame)
        out = cv2.flip(out, flipCode=1)

        # frame = cv2.warpAffine(frame, M, (h, w))
        # print(count)
        img = transforms.ToTensor()(out)  # (3, W, H)
        # print(img.shape)

        img, _ = pad_to_square(img, 0)  # convert to N x N squre matrix, (3, W, W)
        # print(img.shape)
        # Resize
        img = resize(img, opt.img_size)  # (3, M, M)
        # input_imgs = Variable(img.type(Tensor))
        # img = transforms.ToTensor()([img])
        img = img.unsqueeze(0)
        # print(img.size())
        # quit()
        input_imgs = Variable(img)
        with torch.no_grad():
            detections = model(input_imgs)
            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)



        if detections[0] is not None:
            # Rescale boxes to original image
            detections = rescale_boxes(detections[0], opt.img_size, frame.shape[:2])
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                cv2.rectangle(frame, (x1,y1), (x2,y2), (255,33,133),1)



        frame = cv2.resize(frame, frameSize)
        writer.write(frame)
        cv2.imshow('frame', out)
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break
        count += 1
    writer.release()



#
#
# def video_reading():
#
#
#     args = getopt()
#     WAIT = 25
#     path =''
#
#     cap = cv2.VideoCapture()
#     cap.open(path)
#     pps = get_properties(cap) # fps, fourcc, w, h , length
#
#
#     if args.save_video:
#         savepath = ''
#         framesize = ()
#         writer = cv2.VideoWriter(savepath, FOURCC, 20, framesize)
#
#
#     name_desc = tqdm(range(pps[LENGTH]))
#     while 1:
#         ret, frame = cap.read()
#         if ret is False:
#             break
#
#         img = frame
#
#         if args.save_video:
#             writer.write(img)
#         if args.vis:
#             cv2.imshow('11', img)
#             if cv2.waitKey(WAIT) == ord('q'):
#                 break
#
#         name_desc.update(1)
#
#     if args.save_video:
#         writer.release()