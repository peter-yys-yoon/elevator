import argparse


parser = argparse.ArgumentParser(description='PyTorch AlphaPose Training')


"-------------------------------- YOLO v3 ----------------------------------------"
parser.add_argument("--image_folder", type=str,
                    default="data/samples", help="path to dataset")
parser.add_argument("--model_def", type=str,
                    default="/home/peter/workspace/code/yolov3/config/yolov3.cfg", help="path to model definition file")
parser.add_argument("--weights_path", type=str,
                    default="/home/peter/workspace/code/yolov3/weights/yolov3.weights", help="path to weights file")
parser.add_argument("--class_path", type=str,
                    default="/home/peter/workspace/code/yolov3/data/coco.names", help="path to class label file")
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

parser.add_argument('--vis', default=False,
                    action='store_true', help='visualize image')

parser.add_argument('--profile', default=False,
                    action='store_true', help='visualize image')

parser.add_argument('--save_video', dest='save_video',
                    help='whether to save rendered video', default=False, action='store_true')
parser.add_argument('--video', dest='video', help='video-name',
                    default="/home/peter/dataset/gist/org/mid2019/roaming_kdh_trial_1/trim_student1.avi")

opt = parser.parse_args()
