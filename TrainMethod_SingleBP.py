
import numpy as np 
import torch.nn as nn
import torch.optim as optim


from dataSet import CustomizeDataSets
from TrainAndTest import TrainAndTest
from Select_Net import select_net

if __name__ == "__main__":


    ########################### Set Parameters Area Below ####################################

    # BRANCH = 'Master Branch'
    # BRANCH = 'alpha-academic Branch'
    BRANCH = 'alpha-cooperate Branch'
    # BRANCH = 'beta-dev Branch'

    READ_VERSION = 1
    SAVE_VERSION = 1
    TEST_SIZE = 0.3
    RANDOM_SEED = 120
    STEP = 1
    N_DELTA = 1
    SHUFFLE = True


    # the Group in which target BP sited 
    GROUP_ID_List=[
                    'Ki1',
                    # 'Ki2',
                    ]

    # the list of list for setting the BP that will be trained
    BPNAME_ListofList = [
                    ['BP021'],  # BPCASE in Ki1
                    # ['BP008'],                   # BPCASE in Ki2
                        ]
    
    # the dic of each BP
    CHECKPOINT_Dic = {
                        'BP021':None, #  'BP028': None
                        # 'BP033':None, # or 'BP028': ['Ki1', 1, 100]
                        # 'BP043':None,
                        # 'BP008':None,
                        }

    # the dic for using ignore the cases when tran the BPs, 
    # and the ignored cases will not be used as train data 
    # but used for testing and visualizing the performances of the trained models
    EXCEPT_CASE_Dic = {
                        'BP021':['BP021_006','BP021_014','BP021_023','BP021_031'], #  'BP028': None
                        # 'BP033':None, # or 'BP033': ['BP033_006','BP033_014','BP033_023','BP033_031']
                        # 'BP043':None,
                        # 'BP008':None,
                        }

    ########################### Set Parameters Area Above ####################################


    # check the lens of parameters for safety running
    if len(BPNAME_ListofList) != len(GROUP_ID_List):
        print(f'Different lens between variables {GROUP_ID_List} or {BPNAME_ListofList}, please check it!')
    
    else:
        for GROUP, BPNAME_List in zip(GROUP_ID_List, BPNAME_ListofList):
            # Loop for each BP
            for BPNAME in BPNAME_List:
                # extract the check and except information from a dictionary
                CHECKPOINT = CHECKPOINT_Dic[BPNAME]
                EXCEPT_CASE = EXCEPT_CASE_Dic[BPNAME]

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
                                    'EPOCHS' : 6000, # the model will be trained with 6000 epoch
                                    'BATCHSIZES' : 128,
                                    'LOSS_FN' : nn.MSELoss(),
                                    'OPTIMIZER' : optimizer,
                                    'SCHEDULER' : scheduler,
                                    'MODEL_SAVECYCLE' : [
                                                        [2000,500], # the trained model will be Saved periodicaly on Epoch
                                                        [4000,200], # [2000,500] means the model will be saved in each 200 epoch
                                                        [5800,100], # between epoch 2000 and 4000, like 2200 and 2400
                                                        [6000,10],  # please notify that the 4000 can be divided by 200 without remainder
                                                        #[2000,10], # and the last epoch must equal to the value of you set in 'EPOCHS' ,like 6000
                                                        ],
                                    'RECORDER_SAVECYCLE' :[
                                                        [2000,500], # the principle is same as above, 
                                                        [4000,200], # we suggest you keep the value same with 'MODEL_SAVECYCLE',
                                                        [5800,100], # thouhgt the value can be changed in the principle
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

        print('Done!')
    


