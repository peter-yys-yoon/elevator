import os
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
    DATAPATH = '/home/peter/extra/dataset/gist/elevator'
    ENV = 'elev'
    PROJECTPATH = '/home/peter/extra/Workspace/code/elev'
    PYNAME = 'main.py'
    OUTPATH = os.path.join(PROJECTPATH, 'output/alpha')
    "---------------------------------------------------"

    PATH_CONDA = '/home/peter/anaconda3/envs'
    PY = f'{PROJECTPATH}/{PYNAME}'
    EXE = f'{PATH_CONDA}/{ENV}/bin/python'
    os.makedirs(OUTPATH, exist_ok=True)

    # filelist = os.listdir(DATAPATH)
    filelist = get_dy_list()
    filelist.sort()
    print("Total files: ", len(filelist))
    for idx, vname in enumerate(filelist):
        if not vname.endswith('avi'):
            continue
        print('Running on ', vname, idx, '/', len(filelist))
        
        invideo = os.path.join(DATAPATH, vname)
        command = f'{EXE} {PY} --save_video --video {invideo} --outdir {OUTPATH}'
        os.system(command)
        # quit()


runner()