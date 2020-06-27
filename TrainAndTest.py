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
from CNN_models import ConvNet_2

class TrainAndTest():
    def __init__(self, model, dataset, input_folder='../Save/BP028/Step_6/', output_folder='../Save/BP028/Step_6/', 
                    checkpoint=None, read_version=1, save_version=1):
        self.INPUT_FOLDER = input_folder
        self.OUTPUT_FOLDER = output_folder
        if not os.path.exists(os.path.join(self.OUTPUT_FOLDER, 'model')):
            os.makedirs(os.path.join(self.OUTPUT_FOLDER, 'model'))
        if not os.path.exists(os.path.join(self.OUTPUT_FOLDER, 'recorder')):
            os.makedirs(os.path.join(self.OUTPUT_FOLDER, 'recorder'))
        self.MODEL = model
        self.DATASET = dataset
        self.STEP = dataset.STEP
        self.BPNAME = dataset.BPNAME
        self.CHECKPOINT = checkpoint
        self.CHECKBP = None
        self.CHECKSTEP = None
        self.CHECKEPOCH = None
        self.READ_VERSION = read_version
        self.SAVE_VERSION = save_version
        self.TRAIN_PARAMS_DICT = None
        self.TEST_PARAMS_DICT = None
        self.DEVICE = None        

        if torch.cuda.is_available():
            self.DEVICE = torch.device('cuda')
            print(' Runing on the GPU...')
        else:
            self.DEVICE = torch.device('cpu')
            print(' Runing on the CPU...')

        self.MODEL.to(self.DEVICE)

    def _register_checkpoint(self):
        CHECK_EPOCH = int(self.CHECKPOINT[-1])
        model_checkpath = os.path.join(self.INPUT_FOLDER, f"model/model_V{self.READ_VERSION}_epoch_{CHECK_EPOCH}.pt")
        recorder_checkpath = os.path.join(self.INPUT_FOLDER, f'recorder/optim_V{self.READ_VERSION}_epoch_{CHECK_EPOCH}.pt')
        MODEL_CHECK_Dict = torch.load(model_checkpath)
        RECORDER_CHECK_Dict = torch.load(recorder_checkpath)

        MODEL_Dict = MODEL_CHECK_Dict['MODEL']
        MODEL_RECORDER_PD = MODEL_CHECK_Dict['RECORDER']
        OPTIMIZER_Dict = RECORDER_CHECK_Dict['OPTIMIZER']
        SCHEDULER_Dict = RECORDER_CHECK_Dict['SCHEDULER']

        return CHECK_EPOCH, MODEL_Dict, MODEL_RECORDER_PD, OPTIMIZER_Dict, SCHEDULER_Dict


    ############################  Train & Validation  ################################
    def train(self,train_params_Dict):
        self.TRAIN_PARAMS_DICT = train_params_Dict

        #TRAIN_LR = self.TRAIN_PARAMS_DICT['LR']
        TRAIN_EPOCHS = self.TRAIN_PARAMS_DICT['EPOCHS']
        TRAIN_BATCHSIZES = self.TRAIN_PARAMS_DICT['BATCHSIZES']
        TRAIN_LOSS_FN = self.TRAIN_PARAMS_DICT['LOSS_FN']
        TRAIN_LOSS_FN.to(self.DEVICE)
        TRAIN_OPTIMIZER = self.TRAIN_PARAMS_DICT['OPTIMIZER']
        TRAIN_SCHEDULER = self.TRAIN_PARAMS_DICT['SCHEDULER']
        TRAIN_MODEL_SAVECYCLE = self.TRAIN_PARAMS_DICT['MODEL_SAVECYCLE']
        TRAIN_RECORDER_SAVECYCLE = self.TRAIN_PARAMS_DICT['RECORDER_SAVECYCLE']
        TRAIN_NUM_WORKERS = self.TRAIN_PARAMS_DICT['NUM_WORKERS']
        TRAIN_VALIDATION = self.TRAIN_PARAMS_DICT['VALIDATION']
        TRAIN_VERBOSE = self.TRAIN_PARAMS_DICT['VERBOSE']
        TRAIN_TRANSFER = self.TRAIN_PARAMS_DICT['TRANSFER']
        TRAIN_CHECK_OPTIMIZER = self.TRAIN_PARAMS_DICT['CHECK_OPTIMIZER']
        TRAIN_CHECK_SCHEDULER = self.TRAIN_PARAMS_DICT['CHECK_SCHEDULER']

        START_EPOCH = 0
        END_EPOCH = TRAIN_EPOCHS

        RECORDER_PD = pd.DataFrame([['Epoch', 'LR', 'Train Loss', 'Validation Loss']])

        ########################  Checkpoint  ##################################
        if self.CHECKPOINT is not None:
            CHECK_EPOCH, MODEL_Dict, MODEL_RECORDER_PD, OPTIMIZER_Dict, SCHEDULER_Dict = self._register_checkpoint()
            self.MODEL.load_state_dict(MODEL_Dict)
            RECORDER_PD = MODEL_RECORDER_PD

            START_EPOCH = CHECK_EPOCH
            if TRAIN_TRANSFER:
                END_EPOCH = CHECK_EPOCH + TRAIN_EPOCHS
                print(f'从Epoch={CHECK_EPOCH}处进行迁移至新的训练阶段，新阶段将迭代{TRAIN_EPOCHS}个Epochs。')
            else:
                print(f'从Epoch={CHECK_EPOCH}处继续训练，直到Epoch={TRAIN_EPOCHS}')
    
            if TRAIN_CHECK_OPTIMIZER:
                print('加载以往optimizer的参数...')
                TRAIN_OPTIMIZER.load_state_dict(OPTIMIZER_Dict)
            else:
                print('未加载以往optimizer的参数')

            if TRAIN_CHECK_SCHEDULER:
                print('加载以往scheduler的参数...')
                TRAIN_SCHEDULER.load_state_dict(SCHEDULER_Dict)
            else:
                print('未加载以往scheduler的参数')

        ###########################  DataLoader  #######################################
        train_datainfo, traindatasets = self.DATASET.select('train')
        trainloader = Data.DataLoader(dataset=traindatasets, batch_size=TRAIN_BATCHSIZES, 
                                        shuffle=True, num_workers=TRAIN_NUM_WORKERS, pin_memory=True)
        if TRAIN_VALIDATION:
            validation_datainfo, validationdatasets = self.DATASET.select('validation')
            validationloader = Data.DataLoader(dataset=validationdatasets, batch_size=TRAIN_BATCHSIZES, 
                                            shuffle=True, num_workers=TRAIN_NUM_WORKERS, pin_memory=True)

    
        ########################  Epoch Loop  ##########################################
        self.MODEL.train()
        ################  Epoch Clock  #############
        if TRAIN_VERBOSE == 1 or TRAIN_VERBOSE == 2:
            epoch_start = time.time()

        for epoch in range(START_EPOCH+1, END_EPOCH+1):
            #####################  Train Batch Loop  ####################################
            RECORDER_List = []
            RECORDER_List.append(epoch)#记录epoch
            RECORDER_List.append(TRAIN_OPTIMIZER.state_dict()['param_groups'][0]['lr'])#记录LR
            #####  Load Clock  ###### 
            load_start = time.time()
            for batch_id, (X_tensor, Y_tensor) in enumerate(trainloader):
            
                X_input_tensor_gpu = X_tensor.to(self.DEVICE,dtype=torch.float32,non_blocking=True)
                Y_input_tensor_gpu = Y_tensor.to(self.DEVICE,dtype=torch.float32,non_blocking=True)
                ##########  Load&Train Clock  ######
                if TRAIN_VERBOSE == 2:
                    train_start = load_end = time.time()
                    load_timeusage = str(timedelta(seconds=load_end) - timedelta(seconds=load_start))

                self.MODEL.zero_grad()
                Y_output_tensor_gpu = self.MODEL(X_input_tensor_gpu)

                loss = TRAIN_LOSS_FN(Y_output_tensor_gpu,Y_input_tensor_gpu)
                loss.backward()
                TRAIN_OPTIMIZER.step()
                ##########  Train&Load Clock  ######  
                if TRAIN_VERBOSE == 2: 
                    load_start = train_end = time.time()
                    train_timeusage = str(timedelta(seconds=train_end) - timedelta(seconds=train_start))
                    print(f'Name={self.BPNAME}, Step={self.STEP}, Epoch : {epoch}, Batch ID : {batch_id}, Loss ：{loss.item()}, Load timeusage : {load_timeusage}, Train timeusage : {train_timeusage}')

            RECORDER_List.append(loss.item())#记录train loss
            TRAIN_SCHEDULER.step()

            #####################  Validation Batch Loop  #################################
            if TRAIN_VALIDATION:

                self.MODEL.eval()
                with torch.no_grad():
                    #####  Load Clock  ###### 
                    load_start = time.time()
                    for batch_id, (X_tensor, Y_tensor) in enumerate(validationloader):

                        X_input_tensor_gpu = X_tensor.to(self.DEVICE,dtype=torch.float32,non_blocking=True)
                        Y_input_tensor_gpu = Y_tensor.to(self.DEVICE,dtype=torch.float32,non_blocking=True)
                        ##########  Load&Val Clock  ######
                        if TRAIN_VERBOSE == 2:
                            val_start = load_end = time.time()
                            load_timeusage = str(timedelta(seconds=load_end) - timedelta(seconds=load_start))

                        self.MODEL.zero_grad()
                        Y_output_tensor_gpu = self.MODEL(X_input_tensor_gpu)

                        loss = TRAIN_LOSS_FN(Y_output_tensor_gpu,Y_input_tensor_gpu)
                        ##########  Val & Load Clock  ######  
                        if TRAIN_VERBOSE == 2: 
                            load_start = val_end = time.time()
                            val_timeusage = str(timedelta(seconds=val_end) - timedelta(seconds=val_start))
                            print(f'Name={self.BPNAME}, Step={self.STEP}, Epoch : {epoch}, Batch ID : {batch_id}, Loss ：{loss.item()}, Load timeusage : {load_timeusage}, Val timeusage : {val_timeusage}')

                RECORDER_List.append(loss.item())#记录validaion loss
                self.MODEL.train()
            else:
                RECORDER_List.append(None)
            RECORDER_PD = RECORDER_PD.append([RECORDER_List], ignore_index=True)#将本轮epoch的记录存起来
            print(f'Name={self.BPNAME}, Step={self.STEP}, Epoch={epoch}, LR={RECORDER_List[1]}, Train Loss={RECORDER_List[2]}, Validation Loss={RECORDER_List[3]}')

            ################  Epoch Clock  #############    
            if TRAIN_VERBOSE == 1 or TRAIN_VERBOSE == 2:
                epoch_end = time.time()
                epoch_timesuage = str(timedelta(seconds=epoch_end) - timedelta(seconds=epoch_start))
                epoch_start = epoch_end
                print(f'Epoch {epoch} timeusage : {epoch_timesuage}')
                if TRAIN_VERBOSE == 2:
                    print('\n')

            ###############################  Save Cycle  ####################################
            if epoch % TRAIN_MODEL_SAVECYCLE == 0 or epoch == TRAIN_EPOCHS :

                ######################  Save Model  ####################################
                model_state = {
                                'MODEL':self.MODEL.state_dict(),
                                'RECORDER':RECORDER_PD
                                }
                torch.save(model_state, os.path.join(self.OUTPUT_FOLDER, 'model', 
                                                        f'model_V{self.SAVE_VERSION}_epoch_{epoch}.pt'))

                ######################  Save Optimizer&Scheduler  ####################################
                other_state = {
                                'OPTIMIZER':TRAIN_OPTIMIZER.state_dict(),
                                'SCHEDULER':TRAIN_SCHEDULER.state_dict()
                                }
                torch.save(other_state, os.path.join(self.OUTPUT_FOLDER, 'recorder', 
                                                            f'optim_V{self.SAVE_VERSION}_epoch_{epoch}.pt'))
                
                ######################  Save Recorder  ####################################
            if epoch % TRAIN_RECORDER_SAVECYCLE == 0 or epoch == TRAIN_EPOCHS :
                RECORDER_PD.to_csv(os.path.join(self.OUTPUT_FOLDER, 'recorder',
                                                        f'recorder_V{self.SAVE_VERSION}_epoch_{epoch}.csv'),float_format='%.4f')


    def test(self, test_params_Dict, checkpoint, test_info_data=None):

        if not os.path.exists(os.path.join(self.OUTPUT_FOLDER, 'test')):
            os.makedirs(os.path.join(self.OUTPUT_FOLDER, 'test'))
        self.TEST_PARAMS_DICT = test_params_Dict
        self.CHECKPOINT = checkpoint
        if self.CHECKPOINT is not None:
            self.CHECKBP = self.CHECKPOINT[0]
            self.CHECKSTEP = self.CHECKPOINT[1]
            self.CHECKEPOCH = self.CHECKPOINT[2]

        TEST_NUM_WORKERS = self.TEST_PARAMS_DICT['NUM_WORKERS']
        TEST_LOSS_FN = self.TEST_PARAMS_DICT['LOSS_FN']
        ########################################## Register Checkpoint  ####################################
        CHECK_EPOCH, MODEL_Dict, MODEL_RECORDER_PD, OPTIMIZER_Dict, SCHEDULER_Dict = self._register_checkpoint()
        self.MODEL.load_state_dict(MODEL_Dict)
        TEST_recorder_Dict = {}
        TEST_recorder_Dict['EPOCH'] = CHECK_EPOCH
        self.MODEL.to(self.DEVICE)
        TEST_LOSS_FN.to(self.DEVICE)
        #创建test结果的保存文件夹
        if not os.path.exists(os.path.join(self.OUTPUT_FOLDER, 'test', 
                    f"model_V{self.READ_VERSION}_epoch_{CHECK_EPOCH}")):
            os.makedirs(os.path.join(self.OUTPUT_FOLDER, 'test', 
                    f"model_V{self.READ_VERSION}_epoch_{CHECK_EPOCH}"))
        ########################################## Select Test Info&Data  ######################################
        if test_info_data  is not None:
            #TEST_BATCHSIZES = test_info_data['TEST_BATCHSIZES']
            TEST_CASE_LIST = test_info_data['TEST_CASE_LIST']
            testloader = test_info_data['TEST_DATALOADER']
        else:
            test_datainfo, testdatasets = self.DATASET.select('test')
            TEST_BATCHSIZES = test_datainfo['n_sample_eachcase']
            TEST_CASE_LIST = test_datainfo['case_index_List']

            testloader = Data.DataLoader(dataset=testdatasets, batch_size=TEST_BATCHSIZES, 
                                            shuffle=False, num_workers=TEST_NUM_WORKERS, pin_memory=True)

        ########################################  Test Case Loop  #######################################
        self.MODEL.eval()

        with torch.no_grad():
            for case_id, (X_tensor, Y_tensor) in enumerate(testloader):

                X_input_tensor_gpu = X_tensor.to(self.DEVICE,dtype=torch.float32,non_blocking=True)
                Y_input_tensor_gpu = Y_tensor.to(self.DEVICE,dtype=torch.float32,non_blocking=True)

                self.MODEL.zero_grad()
                Y_output_tensor_gpu = self.MODEL(X_input_tensor_gpu)

                loss = TEST_LOSS_FN(Y_output_tensor_gpu,Y_input_tensor_gpu)

                ################### Save & Record ################################
                casename = f'case{TEST_CASE_LIST[case_id]}'
                Y_input_tensor_cpu = Y_input_tensor_gpu.cpu()
                Y_input_Array = Y_input_tensor_cpu.numpy()

                Y_output_tensor_cpu = Y_output_tensor_gpu.cpu()
                Y_output_Array = Y_output_tensor_cpu.numpy()

                savepath = os.path.join(self.OUTPUT_FOLDER, 'test', f"model_V{self.READ_VERSION}_epoch_{CHECK_EPOCH}", f'{casename}.npz')
                np.savez(savepath, input=Y_input_Array, output=Y_output_Array)
                TEST_recorder_Dict[str(casename)] = [loss.item()]
                print(f'Model with [BP={self.CHECKBP}, Step={self.CHECKSTEP}, epoch={CHECK_EPOCH}] test loss in {casename} : {loss.item()}')

        return TEST_recorder_Dict


if __name__ == "__main__":
    print('*** Please run with \'TrainAndTest.py\' script! ***')


                
