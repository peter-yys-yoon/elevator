import os


if __name__ == "__main__":
   
    
    "---------------------HERE-------------------------"    
    DATAPATH = '/home/peter/workspace/dataset/gist/elevator'
    ENV= 'elev'
    PROJECTPATH= '/home/peter/extra/Workspace/code/elev'
    PYNAME = 'yolotest.py'    
    OUTPATH = os.path.join(PROJECTPATH, 'output/yolo')
    "---------------------------------------------------"    

    PATH_CONDA = '/home/peter/anaconda3/envs'
    PY = f'{PROJECTPATH}/{PYNAME}'
    EXE = f'{PATH_CONDA}/{ENV}/bin/python'
    # os.makedirs(OUTPATH, exist_ok=True)

    filelist = os.listdir(DATAPATH)
    filelist.sort()
    print("Total files: ", len(filelist))
    for idx, vname in enumerate(filelist):
        if not vname.endswith('avi'):
            continue
        
        savepath = f'/home/peter/workspace/code/elev/output/yolo/yolo_{vname}'
        if os.path.exists(savepath):
            continue

        "---------------------HERE-------------------------"    

        A_idx = vname.index('A')
        target = vname[A_idx:A_idx+4]
        if target not in [ 'A002']:
            continue

        print('Running on ', vname, idx, '/', len(filelist))
        "---------------------------------------------------"    

        invideo = os.path.join(DATAPATH,vname)
        command = f'{EXE} {PY} --save_video --video {invideo}'
        os.system(command)
        
        
        
        
    
