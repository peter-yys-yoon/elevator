
import os, platform


PUMA = 'puma'
OBAMA = 'obama'

PATH_DATASET = '/home/peter/dataset'
if platform.node() == PUMA:  #
    PATH_DATASET = '/home/peter/dataset'
    PATH_CONDA = '/home/peter/.conda/envs'

else:
    PATH_DATASET = '/home/peter/extra/dataset'
    PATH_CONDA = '/home/peter/anaconda3/envs'



def get_dy_list():
    path = f'/home/peter/extra/dataset/gist/elevator/dyselect/good'
    DATAPATH = '/home/peter/extra/dataset/gist/elevator/video'
    flist = os.listdir(path)
    flist.sort()
    fflist = []
    for f in flist:
        ff = f.split('_')[1]
        fflist.append(os.path.join(DATAPATH,ff))
    return fflist




def runner():
    "---------------------HERE-------------------------"
    DATAPATH = '/home/peter/extra/dataset/gist/elevator/video'
    ENV = 'elev'
    PROJECTPATH = '/home/peter/extra/Workspace/code/elev'
    PYNAME = 'main.py'
    OUTPATH = os.path.join(PROJECTPATH, 'output/paper')
    "---------------------------------------------------"

    PATH_CONDA = '/home/peter/anaconda3/envs'
    PY = f'{PROJECTPATH}/{PYNAME}'
    EXE = f'{PATH_CONDA}/{ENV}/bin/python'
    os.makedirs(OUTPATH, exist_ok=True)

    filelist = os.listdir(DATAPATH)
    # filelist = get_dy_list()
    filelist.sort()
    print("Total files: ", len(filelist))
    for idx, vname in enumerate(filelist):
        if not vname.endswith('avi'):
            continue
        
        if vname[4:8] not in ['A004']:
                continue
            
        print('Running on ', vname, idx, '/', len(filelist))
        
        invideo = os.path.join(DATAPATH, vname)
        command = f'{EXE} {PY} --save_video --video {invideo} --outdir {OUTPATH}'
        os.system(command)
        quit()

def run_yolo():
    "---------------------HERE-------------------------"

    DATAPATH = '/home/peter/workspace/dataset/gist/elevator/video'

    # DATAPATH = '/home/peter/workspace/dataset/aicity/aic20/AIC20_track4/test-data'
    # DATAPATH = '/home/peter/workspace/dataset/aicity/aic19-track3-train-data'
    ENV = 'alpha'
    PROJECTPATH = '/home/peter/workspace/projects/xiah/AlphaPose/yolo'
    PYNAME = 'video_demo.py'
    OUTPATH = '/home/peter/workspace/dataset/gist/elevator/yolotxt'


    # OUTPATH = '/home/peter/workspace/dataset/aicity/train_yolo2'
    # OUTPATH = '/home/peter/workspace/dataset/aicity/aic20/AIC20_track4/test-data-yolo'
    "---------------------------------------------------"

    PY = f'{PROJECTPATH}/{PYNAME}'
    EXE = f'{PATH_CONDA}/{ENV}/bin/python'


    filelist = os.listdir(DATAPATH)
    print("Total files: ", len(filelist))
    for idx, vname in enumerate(filelist):


        # if not vname.endswith('mp4'):
        #     continue
        tmp = vname.split('.')[0]
        result = os.path.join(OUTPATH, f'yolo_{tmp}.txt')

        if os.path.exists(result):
            print('Passed~', vname, idx, '/', len(filelist))
            continue
        "---------------------HERE-------------------------"
        print('Running on ', vname, idx, '/', len(filelist))
        "---------------------------------------------------"

        invideo = os.path.join(DATAPATH, vname)
        command = f'{EXE} {PY} --video {invideo} --outdir {OUTPATH} --save_video'
        # print(command)
        os.system(command)
        # quit()


def run_alpha():
    "---------------------HERE-------------------------"
    DATAPATH = '/home/peter/extra/dataset/gist/elevator/video'
    ENV = 'alpha'
    PROJECTPATH = '/home/peter/workspace/code/AlphaPose'
    PYNAME = 'video_demo.py'
    OUTPATH = '/home/peter/extra/dataset/gist/elevator/alphajson'
    "---------------------------------------------------"

    PATH_CONDA = '/home/peter/anaconda3/envs'
    PY = f'{PROJECTPATH}/{PYNAME}'
    EXE = f'{PATH_CONDA}/{ENV}/bin/python'

    filelist = os.listdir(DATAPATH)
    filelist.sort()
    print("Total files: ", len(filelist))
    for idx, vname in enumerate(filelist[1:]):
        "---------------------HERE-------------------------"
        print('Running on ', vname, idx, '/', len(filelist))


        if vname[4:8] not in ['A004']:
            continue
        "---------------------------------------------------"

        invideo = os.path.join(DATAPATH, vname)
        command = f'{EXE} {PY}  --video {invideo} --outdir {OUTPATH}'
        os.system(command)
        # quit()


def run_trim():
    "---------------------HERE-------------------------"
    DATAPATH = '/home/peter/workspace/dataset/gist/elevator/raw_video'
    ENV = 'elev'
    # PYNAME = '/home/peter/workspace/code/tiah_module/standalone/vtrim.py'
    PYNAME = '/home/peter/workspace/projects/xiah/tiah_module/standalone/vtrim.py'
    OUTPATH = '/home/peter/workspace/dataset/gist/elevator/video'
    "---------------------------------------------------"

    PY = PYNAME
    EXE = f'{PATH_CONDA}/{ENV}/bin/python'
    os.makedirs(OUTPATH, exist_ok=True)

    filelist = os.listdir(DATAPATH)
    filelist.sort()

    print("Total files: ", len(filelist))
    for idx, vname in enumerate(filelist):
        if not vname.endswith('avi'):
            continue

        if os.path.exists(os.path.join(OUTPATH,vname)):
            print('Exist!! ', vname, idx, '/', len(filelist))
            continue
        # if vname[4:8] not in ['A006' , 'A004']:
        #     continue

        print('Running on ', vname, idx, '/', len(filelist))

        invideo = os.path.join(DATAPATH, vname)
        command = f'{EXE} {PY} --video {invideo} --outdir {OUTPATH} --range -1'
        os.system(command)

# runner()
# run_trim()
run_yolo()
# run_alpha()
