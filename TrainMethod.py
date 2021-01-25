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
from tools import data_decorater

if __name__ == "__main__":
        ###################### Initialize Parameters ####################################
    READ_VERSION = 1
    SAVE_VERSION = 1
    TEST_SIZE = 0.3
    RANDOM_SEED = 120
    # BPNAME_List = ['BP028','BP033','BP043']
    # BPNAME_List = ['BP032']
    #STEP_List = [1,3]
    GROUP_ID_List = ['Ki1','Ki2','Ki3','Ki4','Ki5']
    GROUP_ID_List=['Ki1']
    STEP_List = [1]

    CHECKPOINT = None
    #CHECKPOINT = ['Ki1', 1, 99] ###STEP == 1 , EPOCH == 590
    CHECK_EACH_STEP = False
    CHECK_EACH_GROUP = False
    SHUFFLE = True
    N_DELTA = 1

    #提取checkpoint的信息
    if CHECKPOINT is not None:
        START_GROUP = CHECKPOINT[0]
        START_GROUP_INDEX = GROUP_ID_List.index(START_GROUP)

        STRT_STEP = CHECKPOINT[1]
        START_STEP_INDEX = STEP_List.index(STRT_STEP)
    else:
        START_GROUP_INDEX = 0
        START_STEP_INDEX = 0

        
    #根据checkpoint重构循环队列
    for GROUP_ID in GROUP_ID_List[START_GROUP_INDEX:] if isinstance(GROUP_ID_List[START_GROUP_INDEX:],list) else [GROUP_ID_List[START_GROUP_INDEX:]]:
        
        # INFO_path = f'../NpyData/{BPNAME}/_info.npz'
        # INFO_file = np.load(INFO_path)
        # GROUP_ID = INFO_file['GROUP_ID']

        for STEP in STEP_List[START_STEP_INDEX:] if isinstance(STEP_List[START_STEP_INDEX:], list) else [STEP_List[START_STEP_INDEX:]]:

            if CHECKPOINT is not None:
                CHECKPOINT[0] = GROUP_ID
                CHECKPOINT[1] = STEP


            # EXCEPT_BP = ['BP032']
            EXCEPT_CASE = ['BP028_006','BP028_014','BP028_023','BP028_031']
            EXCEPT_BP = None
            ONLY_BP = ['BP022'] #仅仅允许设置1个BP
            EXCEPT_CASE = None

            DATA_FOLDER = f'../Save/alpha-cooperate Branch/TrainData'


            if ONLY_BP is not None:
                INPUT_FOLDER = f'../Save/alpha-cooperate Branch/TrainResults/Step_{int(STEP):02}/{ONLY_BP[0]}'
                OUTPUT_FOLDER = f'../Save/alpha-cooperate Branch/TrainResults/Step_{int(STEP):02}/{ONLY_BP[0]}'
                print(f'BP_ID = {ONLY_BP[0]}, STEP = {int(STEP):02}')
            else:
                INPUT_FOLDER = f'../Save/alpha-cooperate Branch/TrainResults/Step_{int(STEP):02}/{GROUP_ID}'
                OUTPUT_FOLDER = f'../Save/alpha-cooperate Branch/TrainResults/Step_{int(STEP):02}/{GROUP_ID}'
                print(f'GROUP_ID = {GROUP_ID}, STEP = {int(STEP):02}')

            mydataset = CustomizeDataSets(DATA_FOLDER,STEP,GROUP_ID,EXCEPT_BP,ONLY_BP,EXCEPT_CASE,TEST_SIZE,SHUFFLE,RANDOM_SEED)
            #model = ConvNet_2(3+int(STEP/6))
            #add_dem = data_decorater(INFO_path)
            model = select_net(GROUP_ID, 1+3+int(STEP/N_DELTA))
            MyTrainAndTest = TrainAndTest(model,mydataset,INPUT_FOLDER,OUTPUT_FOLDER,None,GROUP_ID,CHECKPOINT,READ_VERSION,SAVE_VERSION)
            ############################## Train Paramters #################################
            LR = 0.0001
            Train_lambda = lambda epoch: 1/np.sqrt(((epoch % 500)+1.0))
            optimizer = optim.Adam(MyTrainAndTest.MODEL.parameters(), lr = LR, weight_decay = 1e-6)
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, Train_lambda)
            TRAIN_PARAMS_DICT = {
                                'EPOCHS' : 1000,
                                'BATCHSIZES' : 128,
                                'LOSS_FN' : nn.MSELoss(),
                                'OPTIMIZER' : optimizer,
                                'SCHEDULER' : scheduler,
                                'MODEL_SAVECYCLE' : [
                                                    [200,100], #前2000 epoch 每过500个epoch保存一下
                                                    [400,50], #前2000-4000 epoch 每过250个epoch保存一下
                                                    [1000,10],
                                                    #[80, 2],
                                                    #[2000,10]
                                                     ],
                                'RECORDER_SAVECYCLE' :[
                                                    [200,100], #前2000 epoch 每过500个epoch保存一下
                                                    [400,50], #前2000-4000 epoch 每过250个epoch保存一下
                                                    #[60,2],
                                                    [1000, 10],
                                                    #[2000,10]
                                                     ],
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
        if not CHECK_EACH_GROUP:
            CHECKPOINT = None
        START_STEP_INDEX = 0
    


