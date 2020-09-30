import os
import torch
import time
import datetime
from datetime import timedelta
import numpy as np 
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from tqdm import tqdm

from dataSet import CustomizeDataSets
from TrainAndTest import TrainAndTest
from Select_Net import select_net

if __name__ == "__main__":
        ###################### Initialize Parameters ####################################
    READ_VERSION = 1
    SAVE_VERSION = 1
    TEST_SIZE = 0.3
    RANDOM_SEED = 120
    BPNAME_List = ['BP028','BP033','BP043']
    BPNAME_List = ['BP032']
    STEP_List = [1,3]
    STEP_List = [1]
    CHECKPOINT = None
    #CHECKPOINT = ['BP028', 1, 590] ###STEP == 1 , EPOCH == 590
    CHECK_EACH_STEP = False
    CHECK_EACH_BP = False
    SHUFFLE = True
    N_DELTA = 1


    #提取checkpoint的信息
    if CHECKPOINT is not None:
        START_BP = CHECKPOINT[0]
        START_BP_INDEX = BPNAME_List.index(START_BP)

        STRT_STEP = CHECKPOINT[1]
        START_STEP_INDEX = STEP_List.index(STRT_STEP)
    else:
        START_BP_INDEX = 0
        START_STEP_INDEX = 0

        
    #根据checkpoint重构循环队列
    for BPNAME in BPNAME_List[START_BP_INDEX:] if isinstance(BPNAME_List[START_BP_INDEX:],list) else [BPNAME_List[START_BP_INDEX:]]:
        
        INFO_path = f'../NpyData/{BPNAME}/_info.npz'
        INFO_file = np.load(INFO_path)
        GROUP_ID = INFO_file['GROUP_ID']

        for STEP in STEP_List[START_STEP_INDEX:] if isinstance(STEP_List[START_STEP_INDEX:], list) else [STEP_List[START_STEP_INDEX:]]:

            if CHECKPOINT is not None:
                CHECKPOINT[0] = BPNAME
                CHECKPOINT[1] = STEP

            INPUT_FOLDER = f'../Save/{BPNAME}/Step_{STEP}/'
            OUTPUT_FOLDER = f'../Save/{BPNAME}/Step_{STEP}/'
            DATA_FOLDER = f'../TrainData/{BPNAME}/Step_{STEP}/'

            print(f'BPNAME = {BPNAME}, STEP = {STEP}')

            mydataset = CustomizeDataSets(DATA_FOLDER,BPNAME,STEP,TEST_SIZE,SHUFFLE,RANDOM_SEED)
            #model = ConvNet_2(3+int(STEP/6))
            model = select_net(GROUP_ID,int(STEP/N_DELTA)+4)
            MyTrainAndTest = TrainAndTest(model, mydataset, INPUT_FOLDER, OUTPUT_FOLDER,
                                            CHECKPOINT, READ_VERSION, SAVE_VERSION)
            ############################## Train Paramters #################################
            LR = 0.0001
            Train_lambda = lambda epoch: 1/np.sqrt(((epoch % 500)+1.0))
            optimizer = optim.Adam(MyTrainAndTest.MODEL.parameters(), lr = LR, weight_decay = 1e-6)
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, Train_lambda)
            TRAIN_PARAMS_DICT = {
                                'EPOCHS' : 2000,
                                'BATCHSIZES' : 128,
                                'LOSS_FN' : nn.MSELoss(),
                                'OPTIMIZER' : optimizer,
                                'SCHEDULER' : scheduler,
                                'MODEL_SAVECYCLE' : 10,
                                'RECORDER_SAVECYCLE' : 100,
                                'NUM_WORKERS' : 3,
                                'VALIDATION' : True,
                                'VERBOSE' : 1,
                                'TRANSFER' : False,
                                'CHECK_OPTIMIZER' : True,
                                'CHECK_SCHEDULER' : True,
                                }
            MyTrainAndTest.train(TRAIN_PARAMS_DICT)
            if not CHECK_EACH_STEP:
                CHECKPOINT = None
        if not CHECK_EACH_BP:
            CHECKPOINT = None
        START_STEP_INDEX = 0
    


