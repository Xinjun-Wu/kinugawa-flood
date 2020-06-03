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

from dataSet import YodogawaDataSets
from TrainAndTest import TrainAndTest
from CNN_models import ConvNet_2

    ###################### Initialize Parameters ####################################
READ_VERSION = 1
SAVE_VERSION = 1
TVT_RATIO = [0.1, 0.1, 0.2]
TEST_SPECIFIC = [10,11]
RANDOM_SEED = 120
BPNAME_List = ['BP028']
STEP_List = [6, 12, 18, 24, 30, 36]
#STEP_List = [6]
CHECKPOINT = None
#CHECKPOINT = [12, 590] ###STEP == 6 , EPOCH == 5
CHECK_EACH_STEP = False
CHECK_EACH_BP = False

if CHECKPOINT is not None:
    START_BP = CHECKPOINT[0]
    START_BP_INDEX = BPNAME_List.index(START_BP)
    BPNAME_List = BPNAME_List[START_BP_INDEX:]
    if not isinstance(BPNAME_List, list):
        BPNAME_List = [BPNAME_List]

    STRT_STEP = CHECKPOINT[1]
    START_STEP_INDEX = STEP_List.index(STRT_STEP)
    STEP_List = STEP_List[START_STEP_INDEX:]
    if not isinstance(STEP_List, list):
        STEP_List = [STEP_List]

BP_ID = 0
for BPNAME in BPNAME_List:
    BP_ID += 1

    STEP_ID = 0
    for STEP in STEP_List:
        STEP_ID += 1
        INPUT_FOLDER = f'../Save/{BPNAME}/Step_{STEP}/'
        OUTPUT_FOLDER = f'../Save/{BPNAME}/Step_{STEP}/'
        Data_FOLDER = f'../TrainData/{BPNAME}/Step_{STEP}/'


        mydataset = YodogawaDataSets(STEP, Data_FOLDER, TVT_RATIO, TEST_SPECIFIC, RANDOM_SEED)
        model = ConvNet_2(3+int(STEP/6))
        MyTrainAndTest = TrainAndTest(model, mydataset, INPUT_FOLDER, OUTPUT_FOLDER,
                                        CHECKPOINT, READ_VERSION, SAVE_VERSION)
        ############################## Train Paramters #################################
        LR = 0.0001
        Train_lambda = lambda epoch: 1/np.sqrt(((epoch % 500)+1.0))
        optimizer = optim.Adam(MyTrainAndTest.MODEL.parameters(), lr = LR, weight_decay = 1e-6)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, Train_lambda)
        TRAIN_PARAMS_DICT = {
                            'EPOCHS' : 2000,
                            'BATCHSIZES' : 144,
                            'LOSS_FN' : nn.L1Loss(),
                            'OPTIMIZER' : optimizer,
                            'SCHEDULER' : scheduler,
                            'MODEL_SAVECYCLE' : 10,
                            'RECORDER_SAVECYCLE' : 100,
                            'NUM_WORKERS' : 3,
                            'VALIDATION' : True,
                            'VERBOSE' : 2,
                            'TRANSFER' : False,
                            'CHECK_OPTIMIZER' : True,
                            'CHECK_SCHEDULER' : True,
                            }
        MyTrainAndTest.train(TRAIN_PARAMS_DICT)
        if CHECK_EACH_STEP :
            CHECKPOINT[1] = STEP_List[STEP_ID]
        else:
            CHECKPOINT = None
    if CHECK_EACH_BP:
        CHECKPOINT[0] = BPNAME_List[BP_ID]
    else:
        CHECKPOINT = None

