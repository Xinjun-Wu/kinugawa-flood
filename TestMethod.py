
import os
import torch
import time
import datetime
from datetime import timedelta
import numpy as np 
import pandas as pd
import torch.nn as nn
from torch.nn.modules.loss import MSELoss
import torch.optim as optim
import torch.utils.data as Data
from tqdm import tqdm

from dataSet import CustomizeDataSets
from TrainAndTest import TrainAndTest
from Select_Net import select_net
#from tools import data_decorater

if __name__ == "__main__":
    ###################### Initialize Parameters ####################################
    READ_VERSION = 1
    SAVE_VERSION = 1
    TEST_NUM_WORKERS = 0
    # GROUP_ID_List = ['Ki1','Ki2','Ki3','Ki4','Ki5']
    GROUP_ID_List = ['Ki1']
    STEP_List = [1]

    START_EPOCH =9990
    END_EPOCH = 10000
    EPOCH_STEP = 10

    CHECKPOINT = None
    SAVE_CYCLE = 10
    TEST_CASE_LIST = ['BP028_006','BP028_014','BP028_023','BP028_031']
    N_DELTA = 1

    for GROUP_ID in GROUP_ID_List:

        # #读取信息描述文件，提取破堤区域数值模拟网格代号GROUP_ID
        # INFO_path = f'../NpyData/{BPNAME}/_info.npz'
        # INFO_file = np.load(INFO_path)
        # GROUP_ID = INFO_file['GROUP_ID']

        for STEP in STEP_List:

            INPUT_FOLDER = f'../Save/Step_{int(STEP):02}/{GROUP_ID}'
            OUTPUT_FOLDER = f'../Save/Step_{int(STEP):02}/{GROUP_ID}'
            DATA_FOLDER = f'../TrainData'

            for CASENAME in TEST_CASE_LIST:
                print(f'######### Testing on case {CASENAME} #########')
                # mydataset = CustomizeDataSets(STEP, Data_FOLDER, TVT_RATIO, TEST_SPECIFIC, RANDOM_SEED, BPNAME)
                # test_datainfo, testdatasets = mydataset.select('test')
                # TEST_BATCHSIZES = test_datainfo['n_sample_eachcase']
                # TEST_CASE_LIST = test_datainfo['case_index_List']
                # testloader = Data.DataLoader(dataset=testdatasets, batch_size=TEST_BATCHSIZES, 
                #                                     shuffle=False, num_workers=TEST_NUM_WORKERS, pin_memory=True)

                #time clock start
                start_clock = time.time()
                #从文件读取数据
                case_path = os.path.join(DATA_FOLDER, f"Step_{int(STEP):02}",f"{GROUP_ID}",f"{str(CASENAME)}.npz")
                case_data = np.load(case_path)
                learning_data = case_data["learning_data"] #返回的是四阶数组
                teacher_data = case_data['teacher_data']
                learning_data = torch.tensor(learning_data)
                teacher_data = torch.tensor(teacher_data)

                # TEST_INFO_DATA = {
                #                     'TEST_CASE_LIST' : TEST_CASE_LIST,
                #                     'TEST_DATALOADER' : testloader
                #                     }
                TEST_PARAMS_DICT = {
                                'TEST_STEP': STEP,
                                'LOSS_FN' : nn.MSELoss(),
                                'CASENAME': CASENAME,
                                'TEST_DATA_X': learning_data,
                                'TEST_DATA_y': teacher_data,
                                'NUM_WORKERS': 0
                                }

                #model = ConvNet_2(3+int(STEP/6))
                # model = select_net(GROUP_ID,int(STEP/N_DELTA)+4)
                # MyTrainAndTest = TrainAndTest(model, None, INPUT_FOLDER, OUTPUT_FOLDER,
                #                                 CHECKPOINT, READ_VERSION, SAVE_VERSION)   
                # add_dem = data_decorater(INFO_path)
                model = select_net(GROUP_ID, 1+3+int(STEP/N_DELTA))
                MyTrainAndTest = TrainAndTest(model, None, INPUT_FOLDER, OUTPUT_FOLDER,GROUP_ID,
                                            CHECKPOINT, READ_VERSION, SAVE_VERSION)

                TEST_LOSS_path = os.path.join(OUTPUT_FOLDER, 'test', f'{CASENAME}.csv')
                # if os.path.exists(TEST_LOSS_path) and :
                #     TEST_LOSS = pd.read_csv(TEST_LOSS_path, index_col=0)
                # else:
                #     TEST_LOSS = pd.DataFrame()
                TEST_LOSS = pd.DataFrame()

                for epoch in range(START_EPOCH, END_EPOCH, EPOCH_STEP):
                    epoch += EPOCH_STEP

                    CHECKPOINT = [GROUP_ID, STEP, epoch]
                    TEST_recorder_Dict = MyTrainAndTest.test(TEST_PARAMS_DICT, CHECKPOINT)
                    TEST_LOSS = TEST_LOSS.append(pd.DataFrame(TEST_recorder_Dict), ignore_index=True)

                    if epoch % SAVE_CYCLE == 0:
                        TEST_LOSS.to_csv(TEST_LOSS_path, float_format='%.4f',index = False)
                TEST_LOSS.to_csv(TEST_LOSS_path, float_format='%.4f',index = False)
                end_clock = time.time()
                timeusage = str(timedelta(seconds=end_clock) - timedelta(seconds=start_clock))
                print(f'Done for {CASENAME} with time usage : {timeusage} . \r\n')

            


