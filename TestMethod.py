
import os
import torch
import time
from datetime import timedelta
import numpy as np 
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
import argparse


from TrainAndTest import TrainAndTest
from Select_Net import select_net
#from tools import data_decorater

if __name__ == "__main__":
    ###################### Initialize Parameters ####################################

    parser = argparse.ArgumentParser()
    parser.add_argument('BPNAME')
    args = parser.parse_args()

    BPNAME = args.BPNAME

    with open('./runingfiles/log.txt', 'a') as f:
        f.write(f'\n{time.ctime()}: {BPNAME} test model start ...')

    # BRANCH = 'Master Branch'
    # BRANCH = 'Academic Branch'
    BRANCH = 'Cooperate Branch'
    # BRANCH = 'Dev Branch'

    READ_VERSION = 1
    SAVE_VERSION = 1
    TEST_NUM_WORKERS = 0
    STEP=1

    START_EPOCH =5000
    END_EPOCH = 6000
    EPOCH_STEP = 10

    CHECKPOINT = None
    SAVE_CYCLE = 10
    TEST_CASE_LIST = ['_006','_014','_023','_031']
    N_DELTA = 1

    try:

        #读取信息描述文件，提取破堤区域数值模拟网格代号GROUP_ID
        INFO_path = f'../Save/{BRANCH}/NpyData/Info/{BPNAME}_info.npz'
        INFO_file = np.load(INFO_path)
        GROUP_ID = INFO_file['GROUP_ID'].item()

        INPUT_FOLDER = f'../Save/{BRANCH}/TrainResults/Step_{int(STEP):02}/{BPNAME}'
        OUTPUT_FOLDER = f'../Save/{BRANCH}/TrainResults/Step_{int(STEP):02}/{BPNAME}'
        TEST_RESULTS_FOLDER = f'../Save/{BRANCH}/TestResults/Step_{int(STEP):02}/{BPNAME}'
        DATA_FOLDER = f'../Save/{BRANCH}/TrainData'

        for index in TEST_CASE_LIST:
            
            CASENAME = BPNAME+index

            print(f'######### Testing on case {CASENAME} #########')
            start_clock = time.time()
            #从文件读取数据
            case_path = os.path.join(DATA_FOLDER, f"Step_{int(STEP):02}",f"{GROUP_ID}",f"{str(CASENAME)}.npz")
            case_data = np.load(case_path)
            learning_data = case_data["learning_data"] #返回的是四阶数组
            teacher_data = case_data['teacher_data']
            learning_data = torch.tensor(learning_data)
            teacher_data = torch.tensor(teacher_data)

            TEST_PARAMS_DICT = {
                            'TEST_STEP': STEP,
                            'LOSS_FN' : nn.MSELoss(),
                            'CASENAME': CASENAME,
                            'TEST_DATA_X': learning_data,
                            'TEST_DATA_y': teacher_data,
                            'NUM_WORKERS': 0
                            }

            model = select_net(GROUP_ID, 1+3+int(STEP/N_DELTA))
            MyTrainAndTest = TrainAndTest(model, None, INPUT_FOLDER, OUTPUT_FOLDER,TEST_RESULTS_FOLDER, GROUP_ID,
                                        CHECKPOINT, READ_VERSION, SAVE_VERSION)

            TEST_LOSS_path = os.path.join(TEST_RESULTS_FOLDER, f'{CASENAME}.csv')

            TEST_LOSS = pd.DataFrame()

            for epoch in tqdm(range(START_EPOCH, END_EPOCH, EPOCH_STEP)):
                epoch += EPOCH_STEP

                CHECKPOINT = [BPNAME, STEP, epoch]
                TEST_recorder_Dict = MyTrainAndTest.test(TEST_PARAMS_DICT, CHECKPOINT)
                TEST_LOSS = TEST_LOSS.append(pd.DataFrame(TEST_recorder_Dict), ignore_index=True)

                if epoch % SAVE_CYCLE == 0:
                    TEST_LOSS.to_csv(TEST_LOSS_path, float_format='%.8f',index = False)
            TEST_LOSS.to_csv(TEST_LOSS_path, float_format='%.8f',index = False)
            end_clock = time.time()
            timeusage = str(timedelta(seconds=end_clock) - timedelta(seconds=start_clock))
            print(f'Done for {CASENAME} with time usage : {timeusage} . \r\n')

        with open('./runingfiles/log.txt', 'a') as f:
            f.write(f'\n{time.ctime()}: {BPNAME} test model end.')

    except Exception as e:
        print(f'{BPNAME} error: {e}')
        with open('./runingfiles/log.txt', 'a') as f:
            f.write(f'\n{time.ctime()}:     ERROR: {e}')


