import argparse
from vars import *

parser = argparse.ArgumentParser(description='PyTorch AlphaPose Training')


"-------------------------------- YOLO v3 ----------------------------------------"
parser.add_argument("--image_folder", type=str,
                    default="data/samples", help="path to dataset")
parser.add_argument("--model_def", type=str,
                    default=PATH_YOLO_CONFIG_FILE, help="path to model definition file")
parser.add_argument("--weights_path", type=str,
                    default=PATH_YOLO_WEIGHT_FILE, help="path to weights file")

parser.add_argument("--class_path", type=str,
                    default=PATH_YOLO_CLASSNAME_FILE, help="path to class label file")
parser.add_argument("--conf_thres", type=float, default=0.8,
                    help="object confidence threshold")
parser.add_argument("--nms_thres", type=float, default=0.4,
                    help="iou thresshold for non-maximum suppression")
parser.add_argument("--batch_size", type=int, default=1,
                    help="size of the batches")
parser.add_argument("--n_cpu", type=int, default=0,
                    help="number of cpu threads to use during batch generation")
parser.add_argument("--img_size", type=int, default=416,
                    help="size of each image dimension")
parser.add_argument("--checkpoint_model", type=str,
                    help="path to checkpoint model")

"-------------------------------- TIAH ----------------------------------------"

parser.add_argument('--vis', default=False,                    action='store_true', help='visualize image')

parser.add_argument('--profile', default=False, action='store_true', help='visualize image')
parser.add_argument('--wait', default=1, action='store', help='cv2.wait time')

parser.add_argument('--save_video', dest='save_video',
                    help='whether to save rendered video', default=False, action='store_true')
# parser.add_argument('--video', dest='video', help='video-name',default='/home/peter/workspace/dataset/gist/elevator/video/C001A002D001P0002T0001.avi')
parser.add_argument('--video', dest='video', help='video-name',default='/home/peter/workspace/dataset/gist/elevator/video/C001A002D001P0002T0001.avi')
parser.add_argument('--outdir', dest='outdir', help='video-name',default='./output')

opt = parser.parse_args()
