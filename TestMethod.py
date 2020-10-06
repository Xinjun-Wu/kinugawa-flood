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
from tools import data_decorater

if __name__ == "__main__":
    ###################### Initialize Parameters ####################################
    READ_VERSION = 1
    SAVE_VERSION = 1
    TEST_NUM_WORKERS = 0
    BPNAME_List = ['BP028']
    BPNAME_List = ['BP032']
    STEP_List = [1,3]
    STEP_List = [1]
    START_EPOCH =1000
    END_EPOCH = 1600
    EPOCH_STEP = 10
    CHECKPOINT = None
    SAVE_CYCLE = 10
    TEST_CASE_LIST = [2,6]
    TEST_CASE_LIST = np.linspace(1,31,31,True,dtype=np.int)
    TEST_CASE_LIST = [1]
    #CASENAME = 6 #进行测试的case名称
    N_DELTA = 1

    for BPNAME in BPNAME_List:

        #读取信息描述文件，提取破堤区域数值模拟网格代号GROUP_ID
        INFO_path = f'../NpyData/{BPNAME}/_info.npz'
        INFO_file = np.load(INFO_path)
        GROUP_ID = INFO_file['GROUP_ID']

        for STEP in STEP_List:

            INPUT_FOLDER = f'../Save/{BPNAME}/Step_{STEP}/'
            OUTPUT_FOLDER = f'../Save/{BPNAME}/Step_{STEP}/'
            Data_FOLDER = f'../TrainData/{BPNAME}/Step_{STEP}/'

            for CASENAME in TEST_CASE_LIST:
                print(f'######### Testing on case {CASENAME} !#########')
                # mydataset = CustomizeDataSets(STEP, Data_FOLDER, TVT_RATIO, TEST_SPECIFIC, RANDOM_SEED, BPNAME)
                # test_datainfo, testdatasets = mydataset.select('test')
                # TEST_BATCHSIZES = test_datainfo['n_sample_eachcase']
                # TEST_CASE_LIST = test_datainfo['case_index_List']
                # testloader = Data.DataLoader(dataset=testdatasets, batch_size=TEST_BATCHSIZES, 
                #                                     shuffle=False, num_workers=TEST_NUM_WORKERS, pin_memory=True)
                #从文件读取数据
                case_path = os.path.join(Data_FOLDER, f"{str(CASENAME)}.npz")
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
                add_dem = data_decorater(INFO_path)
                model = select_net(GROUP_ID, add_dem.n_add_channel+3+int(STEP/N_DELTA))
                MyTrainAndTest = TrainAndTest(model, None, add_dem, INPUT_FOLDER, OUTPUT_FOLDER,
                                            CHECKPOINT, READ_VERSION, SAVE_VERSION)

                TEST_LOSS_path = os.path.join(OUTPUT_FOLDER, 'test', 
                                f'case{CASENAME} test loss.csv')
                # if os.path.exists(TEST_LOSS_path) and :
                #     TEST_LOSS = pd.read_csv(TEST_LOSS_path, index_col=0)
                # else:
                #     TEST_LOSS = pd.DataFrame()
                TEST_LOSS = pd.DataFrame()

                for epoch in range(START_EPOCH, END_EPOCH, EPOCH_STEP):
                    epoch += EPOCH_STEP

                    CHECKPOINT = [BPNAME, STEP, epoch]
                    TEST_recorder_Dict = MyTrainAndTest.test(TEST_PARAMS_DICT, CHECKPOINT)
                    TEST_LOSS = TEST_LOSS.append(pd.DataFrame(TEST_recorder_Dict), ignore_index=True)

                    if epoch % SAVE_CYCLE == 0:
                        TEST_LOSS.to_csv(TEST_LOSS_path, float_format='%.4f',index = False)
                TEST_LOSS.to_csv(TEST_LOSS_path, float_format='%.4f',index = False)
                print('Done!')

            


