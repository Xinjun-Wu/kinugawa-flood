
import numpy as np 
import torch.nn as nn
import torch.optim as optim
import argparse
import time


from dataSet import CustomizeDataSets
from TrainAndTest import TrainAndTest
from Select_Net import select_net


if __name__ == "__main__":


    ########################### Set Parameters Area Below ####################################

    
    parser = argparse.ArgumentParser()
    #parser.add_argument('GROUP')
    parser.add_argument('BPNAME')
    parser.add_argument('BATCHSIZE')
    # parser.add_argument('EPOCH')
    parser.add_argument('CHECKEPOCH')
    args = parser.parse_args()

    #GROUP= args.GROUP
    BPNAME = args.BPNAME
    BATCHSIZE = int(args.BATCHSIZE)
    # EPOCH = int(args.EPOCH)
    CHECKEPOCH = int(args.CHECKEPOCH)

    print(f'{BPNAME} train start: {time.ctime()}\r\n')
    
    BRANCH = 'Master Branch'
    # BRANCH = 'Academic Branch'
    # BRANCH = 'Cooperate Branch'
    # BRANCH = 'Dev Branch'

    READ_VERSION = 1
    SAVE_VERSION = 1
    TEST_SIZE = 0.3
    RANDOM_SEED = 120
    STEP = 1
    N_DELTA = 1
    SHUFFLE = True
    EPOCH = 6000
    except_case_index = [6,14,23,31]

    #读取信息描述文件，提取破堤区域数值模拟网格代号GROUP_ID
    INFO_path = f'../Save/{BRANCH}/NpyData/Info/{BPNAME[:5]}_info.npz'
    INFO_file = np.load(INFO_path)
    GROUP = INFO_file['GROUP_ID']


    EXCEPT_CASE = [f'{BPNAME}_{x:03}' for x in except_case_index] # BP028_006, BP028_014

    # update the value of check point
    CHECKPOINT = None
    if CHECKEPOCH != 0:
        CHECKPOINT = [GROUP, STEP, CHECKEPOCH]


    ############################################ Set Parameters Area Below ####################################
    # initial the paths 
    DATA_FOLDER = f'../Save/{BRANCH}/TrainData'
    INPUT_FOLDER = f'../Save/{BRANCH}/TrainResults/Step_{int(STEP):02}/{BPNAME}'
    OUTPUT_FOLDER = f'../Save/{BRANCH}/TrainResults/Step_{int(STEP):02}/{BPNAME}'

    ############################################ Set Parameters Area Above ####################################

    mydataset = CustomizeDataSets(DATA_FOLDER,STEP,GROUP,None,
                                    [BPNAME],EXCEPT_CASE,
                                    TEST_SIZE,SHUFFLE,RANDOM_SEED)
    model = select_net(GROUP, 1+3+int(STEP/N_DELTA))

    MyTrainAndTest = TrainAndTest(model,mydataset,INPUT_FOLDER,OUTPUT_FOLDER,None,
                                    GROUP,CHECKPOINT,READ_VERSION,SAVE_VERSION)

    ############################## Set Parameters Area Below #################################
    LR = 0.0001
    Train_lambda = lambda epoch: 1/np.sqrt(((epoch % 500)+1.0))
    optimizer = optim.Adam(MyTrainAndTest.MODEL.parameters(), lr = LR, weight_decay = 1e-6)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, Train_lambda)
    TRAIN_PARAMS_DICT = {
                        'EPOCHS' : EPOCH, # the model will be trained with 6000 epoch
                        'BATCHSIZES' : BATCHSIZE,
                        'LOSS_FN' : nn.MSELoss(),
                        'OPTIMIZER' : optimizer,
                        'SCHEDULER' : scheduler,
                        'MODEL_SAVECYCLE' : [
                                            [2000,500], # the trained model will be Saved periodicaly on Epoch
                                            [4000,200], # [2000,500] means the model will be saved in each 200 epoch
                                            [5000,100], # between epoch 2000 and 4000, like 2200 and 2400
                                            [6000,10],  # please notify that the 4000 can be divided by 200 without remainder
                                            #[2000,10], # and the last epoch must equal to the value of you set in 'EPOCHS' ,like 6000
                                            ],
                        'RECORDER_SAVECYCLE' :[
                                            [2000,500], # the principle is same as above, 
                                            [4000,200], # we suggest you keep the value same with 'MODEL_SAVECYCLE',
                                            [5000,100], # thouhgt the value can be changed in the principle
                                            [6000,10],  # 
                                            #[2000,10], # 
                                            ],
                        'NUM_WORKERS' : 3,
                        'VALIDATION' : True,
                        'VERBOSE' : 1,
                        'TRANSFER' : False,
                        'CHECK_OPTIMIZER' : True,
                        'CHECK_SCHEDULER' : True,
                        }
    ############################## Set Parameters Area Above #################################

    MyTrainAndTest.train(TRAIN_PARAMS_DICT)

    print(f'{BPNAME} train end: {time.ctime()}\r\n')

    


