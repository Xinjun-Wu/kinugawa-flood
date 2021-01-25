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
    TVT_RATIO = [0.5, 0.3, 0.2]
    TEST_SPECIFIC = [10,12]
    RANDOM_SEED = 120
    BPNAME_List = ['BP028','BP033','BP043']
    BPNAME_List = ['BP028']
    STEP_List = [6, 12, 18, 24, 30, 36]
    STEP_List = [6]
    CHECKPOINT = None
    #CHECKPOINT = ['BP033', 12, 590] ###STEP == 6 , EPOCH == 5
    CHECK_EACH_STEP = False
    CHECK_EACH_BP = False

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
        
        INFO_path = f'../Save/Master Branch/NpyData/{BPNAME}/_info.npz'
        INFO_file = np.load(INFO_path)
        GROUP_ID = INFO_file['GROUP_ID']

        for STEP in STEP_List[START_STEP_INDEX:] if isinstance(STEP_List[START_STEP_INDEX:], list) else [STEP_List[START_STEP_INDEX:]]:

            if CHECKPOINT is not None:
                CHECKPOINT[0] = BPNAME
                CHECKPOINT[1] = STEP

            INPUT_FOLDER = f'../Save/Master Branch/TrainResults/{BPNAME}/Step_{STEP}/'
            OUTPUT_FOLDER = f'../Save/Master Branch/TrainResults/{BPNAME}/Step_{STEP}/'
            Data_FOLDER = f'../Save/Master Branch/TrainData/{BPNAME}/Step_{STEP}/'

            print(f'BPNAME = {BPNAME}, STEP = {STEP}')

            mydataset = CustomizeDataSets(STEP, Data_FOLDER, TVT_RATIO, TEST_SPECIFIC, RANDOM_SEED, BPNAME)
            #model = ConvNet_2(3+int(STEP/6))
            model = select_net(GROUP_ID,int(STEP/6)+3)
            MyTrainAndTest = TrainAndTest(model, mydataset, INPUT_FOLDER, OUTPUT_FOLDER,
                                            CHECKPOINT, READ_VERSION, SAVE_VERSION)
            ############################## Train Paramters #################################
            LR = 0.0001
            Train_lambda = lambda epoch: 1/np.sqrt(((epoch % 500)+1.0))
            optimizer = optim.Adam(MyTrainAndTest.MODEL.parameters(), lr = LR, weight_decay = 1e-6)
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, Train_lambda)
            TRAIN_PARAMS_DICT = {
                                'EPOCHS' : 20,
                                'BATCHSIZES' : 36,
                                'LOSS_FN' : nn.L1Loss(),
                                'OPTIMIZER' : optimizer,
                                'SCHEDULER' : scheduler,
                                'MODEL_SAVECYCLE' : 2,
                                'RECORDER_SAVECYCLE' : 2,
                                'NUM_WORKERS' : 3,
                                'VALIDATION' : True,
                                'VERBOSE' : 2,
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
    


