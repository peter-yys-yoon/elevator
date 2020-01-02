from os.path import join


PROJECT_PATH = '/home/peter/workspace/code/elev'

PATH_YOLO_PROJECT= join(PROJECT_PATH,'yolo')
PATH_YOLO_CONFIG_FILE = join(PATH_YOLO_PROJECT,'config/yolov3.cfg')
PATH_YOLO_WEIGHT_FILE= join(PATH_YOLO_PROJECT, 'weights/yolov3.weights')
PATH_YOLO_CLASSNAME_FILE= join(PATH_YOLO_PROJECT, 'data/coco.names')

PATH_DATA_VIDEO =join(PROJECT_PATH,'data/video')
PATH_DATA_YOLOTXT = join(PROJECT_PATH,'data/yolotxt')
PATH_DATA_JSONJSON = join(PROJECT_PATH,'data/alphajson')
PATH_OUTPUT = join(PROJECT_PATH,'data/output')



"-----------------------------------------------"

FALL_THRESHOLD = 120