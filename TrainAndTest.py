import os
from numpy.lib.function_base import average
import torch
import time
import datetime
from datetime import timedelta
import numpy as np 
import pandas as pd
from torch._C import device
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from tqdm import tqdm
from dataSet import CustomizeDataSets
import torch.nn.functional as F

from tools import area_extract

class TrainAndTest():
    def __init__(self, model, dataset, input_folder='../Save/Step_01/Academic', 
                    output_folder='../Save/Step_01/Academic', 
                    test_folder = '../Save/Step_01/Academic',
                    group_id='Ki1',
                    checkpoint=None, read_version=1, save_version=1):
        self.MODEL = model
        self.DATASET = dataset
        self.INPUT_FOLDER = input_folder
        self.OUTPUT_FOLDER = output_folder
        self.TEST_RESULTS_FOLDER = test_folder
        self.GROUP_ID = group_id

        if not os.path.exists(os.path.join(self.OUTPUT_FOLDER,'model')):
            os.makedirs(os.path.join(self.OUTPUT_FOLDER, 'model'))
        if not os.path.exists(os.path.join(self.OUTPUT_FOLDER, 'recorder')):
            os.makedirs(os.path.join(self.OUTPUT_FOLDER, 'recorder'))

        if dataset is not None:
            self.GROUP_ID = dataset.GROUP_ID
            self.STEP = dataset.STEP
            
        #self.ADD_DATA = add_data

        #self.n_add_channel = add_data.n_add_channel
        self.CHECKPOINT = checkpoint
        self.CHECK_GROUP_ID = None
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

    def _register_checkpoint(self, recorder = True):
        #注册检查点，读取检查点信息
        CHECK_EPOCH = int(self.CHECKPOINT[-1])
        model_checkpath = os.path.join(self.INPUT_FOLDER, f"model/model_V{self.READ_VERSION}_epoch_{CHECK_EPOCH}.pt")
        recorder_checkpath = os.path.join(self.INPUT_FOLDER, f'recorder/optim_V{self.READ_VERSION}_epoch_{CHECK_EPOCH}.pt')

        MODEL_CHECK_Dict = torch.load(model_checkpath)
        MODEL_Dict = MODEL_CHECK_Dict['MODEL']
        MODEL_RECORDER_DIC = MODEL_CHECK_Dict['RECORDER']

        if recorder:

            RECORDER_CHECK_Dict = torch.load(recorder_checkpath)
            OPTIMIZER_Dict = RECORDER_CHECK_Dict['OPTIMIZER']
            SCHEDULER_Dict = RECORDER_CHECK_Dict['SCHEDULER']

            return CHECK_EPOCH, MODEL_Dict, MODEL_RECORDER_DIC, OPTIMIZER_Dict, SCHEDULER_Dict

        else:
            OPTIMIZER_Dict = None
            SCHEDULER_Dict = None

            return CHECK_EPOCH, MODEL_Dict

    def _epoch_save_cycle(self,epoch,save_cycle):
        if isinstance(save_cycle, int):
            return_value = save_cycle
            return return_value

        elif isinstance(save_cycle, list):
            return_value = 10 
            for e in save_cycle:
                if epoch <= e[0]:
                    return_value = e[1]
                    return return_value
                else:
                    pass
            return return_value

        else:
            raise ValueError(f'Illegal value of {save_cycle}')

    ############################  Train & Validation  ################################
    def train(self,train_params_Dict):
        self.TRAIN_PARAMS_DICT = train_params_Dict
        #加载训练参数
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
        
        #创建个字典保存训练时的结果
        RECORDER_DIC = {} 
        for header in ['Epoch', 'LR', 'Train Loss', 'Validation Loss']:
            RECORDER_DIC[header] = []
       
        ########################  Checkpoint  ##################################
        if self.CHECKPOINT is not None:
            CHECK_EPOCH, MODEL_Dict, MODEL_RECORDER_DIC, OPTIMIZER_Dict, SCHEDULER_Dict = self._register_checkpoint()
            self.MODEL.load_state_dict(MODEL_Dict)
            if isinstance(MODEL_RECORDER_DIC, dict):
                RECORDER_DIC = MODEL_RECORDER_DIC

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
        traindatasets = self.DATASET.select('train')
        trainloader = Data.DataLoader(dataset=traindatasets, batch_size=TRAIN_BATCHSIZES, 
                                        shuffle=True, num_workers=TRAIN_NUM_WORKERS, pin_memory=True)
        if TRAIN_VALIDATION:
            validationdatasets = self.DATASET.select('test')
            validationloader = Data.DataLoader(dataset=validationdatasets, batch_size=TRAIN_BATCHSIZES, 
                                            shuffle=True, num_workers=TRAIN_NUM_WORKERS, pin_memory=True)
    
        ########################  Epoch Loop  ##########################################
        self.MODEL.train()
        ################  Epoch Clock  #############
        if TRAIN_VERBOSE == 1 or TRAIN_VERBOSE == 2:
            epoch_start = time.time()

        for epoch in range(START_EPOCH+1, END_EPOCH+1):
            #####################  Train Batch Loop  ####################################
            # RECORDER_List = []
            # RECORDER_List.append(epoch)#记录epoch
            # RECORDER_List.append(TRAIN_OPTIMIZER.state_dict()['param_groups'][0]['lr'])
            RECORDER_DIC['Epoch'].append(epoch)
            lr = TRAIN_OPTIMIZER.state_dict()['param_groups'][0]['lr']
            #RECORDER_DIC['LR'].append(TRAIN_OPTIMIZER.state_dict()['param_groups'][0]['lr'])#记录LR
            RECORDER_DIC['LR'].append(lr)
            #####  Load Clock  ###### 
            load_start = time.time()
            for batch_id, (X_tensor, Y_tensor) in enumerate(trainloader):

                # #以添加通道的方式添加其他固定不点的数据，例如DEM，或者梯度
                # if self.ADD_DATA is not None:
                #     X_tensor = self.ADD_DATA.add_dem(X_tensor)
            
                X_input_tensor_gpu = X_tensor.to(self.DEVICE,dtype=torch.float32,non_blocking=True)
                Y_input_tensor_gpu = Y_tensor.to(self.DEVICE,dtype=torch.float32,non_blocking=True)
                ##########  Load&Train Clock  ######
                if TRAIN_VERBOSE == 2:
                    train_start = load_end = time.time()
                    load_timeusage = str(timedelta(seconds=load_end) - timedelta(seconds=load_start))

                self.MODEL.zero_grad()
                Y_output_tensor_gpu = self.MODEL(X_input_tensor_gpu)

                train_loss = TRAIN_LOSS_FN(Y_output_tensor_gpu,Y_input_tensor_gpu)

                train_loss.backward()
                TRAIN_OPTIMIZER.step()

                ##########  Train&Load Clock  ######  
                if TRAIN_VERBOSE == 2: 
                    load_start = train_end = time.time()
                    train_timeusage = str(timedelta(seconds=train_end) - timedelta(seconds=train_start))

                    print(f'Group={self.GROUP_ID}, Step={int(self.STEP):02}, Epoch : {epoch}, Batch ID : {batch_id}, \r\n train_Loss ：{train_loss.item()}, Load timeusage : {load_timeusage}, Train timeusage : {train_timeusage}')

            # RECORDER_List.append(loss.item())#记录train loss
            RECORDER_DIC['Train Loss'].append(train_loss.item())
            TRAIN_SCHEDULER.step()

            #####################  Validation Batch Loop  #################################
            if TRAIN_VALIDATION:

                self.MODEL.eval()
                with torch.no_grad():
                    #####  Load Clock  ###### 
                    load_start = time.time()
                    for batch_id, (X_tensor, Y_tensor) in enumerate(validationloader):

                        # #以添加通道的方式添加其他固定不点的数据，例如DEM，或者梯度
                        # if self.ADD_DATA is not None:
                        #     X_tensor = self.ADD_DATA.add_dem(X_tensor)

                        X_input_tensor_gpu = X_tensor.to(self.DEVICE,dtype=torch.float32,non_blocking=True)
                        Y_input_tensor_gpu = Y_tensor.to(self.DEVICE,dtype=torch.float32,non_blocking=True)
                        ##########  Load&Val Clock  ######
                        if TRAIN_VERBOSE == 2:
                            val_start = load_end = time.time()
                            load_timeusage = str(timedelta(seconds=load_end) - timedelta(seconds=load_start))

                        self.MODEL.zero_grad()
                        Y_output_tensor_gpu = self.MODEL(X_input_tensor_gpu)

                        val_loss = TRAIN_LOSS_FN(Y_output_tensor_gpu,Y_input_tensor_gpu)
                        ##########  Val & Load Clock  ######  
                        if TRAIN_VERBOSE == 2: 
                            load_start = val_end = time.time()
                            val_timeusage = str(timedelta(seconds=val_end) - timedelta(seconds=val_start))
                            print(f'Group={self.GROUP_ID}, Step={int(self.STEP):02}, Epoch : {epoch}, Batch ID : {batch_id}, \r\n val_Loss ：{val_loss.item()}, Load timeusage : {load_timeusage}, Val timeusage : {val_timeusage}')

                # RECORDER_List.append(loss.item())#记录validaion loss
                RECORDER_DIC['Validation Loss'].append(val_loss.item())
        
                self.MODEL.train()
            else:
                #RECORDER_List.append(None)
                RECORDER_DIC['Validation Loss'].append(None)
            # RECORDER_PD = RECORDER_PD.append([RECORDER_List], ignore_index=True)#将本轮epoch的记录存起来

            print(f'Group={self.GROUP_ID}, Step={int(self.STEP):02}, Epoch={epoch}, LR={lr}, Train Loss={train_loss}, Validation Loss={val_loss}')
            
            ################  Epoch Clock  #############    
            if TRAIN_VERBOSE == 1 or TRAIN_VERBOSE == 2:
                epoch_end = time.time()
                epoch_timesuage = str(timedelta(seconds=epoch_end) - timedelta(seconds=epoch_start))
                epoch_start = epoch_end
                print(f'Epoch {epoch} timeusage : {epoch_timesuage}')
                if TRAIN_VERBOSE == 2:
                    print('\n')

            ###############################  Save Cycle  ####################################
            model_cycle = self._epoch_save_cycle(epoch,TRAIN_MODEL_SAVECYCLE)
            if epoch % model_cycle == 0 or epoch == TRAIN_EPOCHS :

                ######################  Save Model  ####################################
                model_state = {
                                'MODEL':self.MODEL.state_dict(),
                                'RECORDER':RECORDER_DIC
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

            recorder_cycle = self._epoch_save_cycle(epoch,TRAIN_RECORDER_SAVECYCLE)
            if epoch % recorder_cycle == 0 or epoch == TRAIN_EPOCHS :
                RECORDER_PD = pd.DataFrame(RECORDER_DIC)
                RECORDER_PD.to_csv(os.path.join(self.OUTPUT_FOLDER, 'recorder',
                        f'recorder_V{self.SAVE_VERSION}_epoch_{epoch}.csv'),float_format='%.8f',index = False)


    def test(self, test_params_Dict, checkpoint):

        if not os.path.exists(self.TEST_RESULTS_FOLDER):
            os.makedirs(self.TEST_RESULTS_FOLDER)
        self.TEST_PARAMS_DICT = test_params_Dict
        self.CHECKPOINT = checkpoint
        if self.CHECKPOINT is not None:
            self.CHECK_GROUP_ID = self.CHECKPOINT[0]
            self.CHECKSTEP = self.CHECKPOINT[1]
            self.CHECKEPOCH = self.CHECKPOINT[2]

        TEST_STEP = self.TEST_PARAMS_DICT['TEST_STEP']
        TEST_LOSS_FN = self.TEST_PARAMS_DICT['LOSS_FN']
        TEST_CASE_NAME = self.TEST_PARAMS_DICT['CASENAME']
        TEST_DATA_X = self.TEST_PARAMS_DICT['TEST_DATA_X']
        TEST_DATA_y = self.TEST_PARAMS_DICT['TEST_DATA_y']
        ########################################## Register Checkpoint  ####################################
        CHECK_EPOCH, MODEL_Dict = self._register_checkpoint(False)
        self.MODEL.load_state_dict(MODEL_Dict)
        TEST_recorder_Dict = {}
        TEST_recorder_Dict['EPOCH'] = [CHECK_EPOCH]
        self.MODEL.to(self.DEVICE)
        TEST_LOSS_FN.to(self.DEVICE)
        #创建test结果的保存文件夹
        if not os.path.exists(os.path.join(self.TEST_RESULTS_FOLDER, 
                    f"model_V{self.READ_VERSION}_epoch_{CHECK_EPOCH}")):
            os.makedirs(os.path.join(self.TEST_RESULTS_FOLDER, 
                    f"model_V{self.READ_VERSION}_epoch_{CHECK_EPOCH}"))
        ########################################## Select Test Info&Data  ######################################
        # if test_info_data  is not None:
        #     #TEST_BATCHSIZES = test_info_data['TEST_BATCHSIZES']
        #     TEST_CASE_LIST = test_info_data['TEST_CASE_LIST']
        #     testloader = test_info_data['TEST_DATALOADER']
        # else:
        #     test_datainfo, testdatasets = self.DATASET.select('test')
        #     TEST_BATCHSIZES = test_datainfo['n_sample_eachcase']
        #     TEST_CASE_LIST = test_datainfo['case_index_List']

        #     testloader = Data.DataLoader(dataset=testdatasets, batch_size=TEST_BATCHSIZES, 
        #                                     shuffle=False, num_workers=TEST_NUM_WORKERS, pin_memory=True)

        ########################################  Test Case Loop  #######################################
        self.MODEL.eval()

        with torch.no_grad():

            Y_output_tensor_gpu_list = []#保存输出的Y
            LOSS_ITEM = []# 保存每次loss

            for sample_id, (X_Input, Y_Input) in enumerate(zip(TEST_DATA_X,TEST_DATA_y)):

                X_tensor = X_Input.unsqueeze(0)
                Y_tensor = Y_Input.unsqueeze(0)

                # #以添加通道的方式添加其他固定不点的数据，例如DEM，或者梯度
                # if self.ADD_DATA is not None:
                #     X_tensor = self.ADD_DATA.add_dem(X_tensor)

                X_input_tensor_gpu = X_tensor.to(self.DEVICE,dtype=torch.float32,non_blocking=True)
                Y_input_tensor_gpu = Y_tensor.to(self.DEVICE,dtype=torch.float32,non_blocking=True)

                if len(Y_output_tensor_gpu_list ) > TEST_STEP-1 :#当保存的Y的个数大于或等于STEP时，使用预测的结果作为下一个周期的输入
                    X_input_tensor_gpu[:,1:1+3,:,:] = Y_output_tensor_gpu_list[sample_id - TEST_STEP]

                self.MODEL.zero_grad()
                Y_output_tensor_gpu = self.MODEL(X_input_tensor_gpu)

                Y_output_array = Y_output_tensor_gpu.cpu()
                Y_output_array = Y_output_array.numpy()

                #剔除水深值小于0的值
                Y_output_tensor_gpu = torch.tensor(Y_output_array,device = self.DEVICE)
                Y_output_tensor_gpu[:,0,:,:] = F.relu(Y_output_tensor_gpu[:,0,:,:])
                
                #利用buffered输入的范围的mask来提取输出的范围
                if sample_id == 0: 
                    # 初始时刻使用破堤点入流作为目标区域生成 buffered mask
                    target_area = X_input_tensor_gpu[0,1+3]
                     
                else:
                    # 后续时刻使用上一时刻的洪水水深范围作为目标区域生成buffered mask
                    target_area = X_input_tensor_gpu[0,1]

                # for n in range(3):
                #     Y_output_array[0,n] = area_extract(target_area,
                #                                     Y_output_tensor_gpu[0,n],10,25,None,0.0001)
                #利用上一时刻的buffered范围提取现在时刻的水深范围
                Y_output_array[0,0] = area_extract(target_area,Y_output_tensor_gpu[0,0],30,30,None,0.0001)
                #利用本时刻的水深范围提取现在时刻的流速计算范围
                for n in [1,2]:
                    Y_output_array[0,n] = area_extract(Y_output_array[0,0],Y_output_tensor_gpu[0,n],0,0,None,0.0001)

                Y_output_tensor_gpu = torch.tensor(Y_output_array,device = self.DEVICE)

                Y_output_tensor_gpu_list.append(Y_output_tensor_gpu)

                loss = TEST_LOSS_FN(Y_output_tensor_gpu,Y_input_tensor_gpu)
                #记录损失值
                LOSS_ITEM.append(loss.item())
                TEST_recorder_Dict[f't_{sample_id + self.CHECKSTEP}'] = [loss.item()]

            ################### Save & Record ################################
            #casename = f'case{TEST_CASE_LIST[case_id]}'
            # Y_input_tensor_cpu = Y_input_tensor_gpu.cpu()
            # Y_input_Array = Y_input_tensor_cpu.numpy()
            Y_input_Array = TEST_DATA_y

            # Y_output_tensor_cpu = Y_output_tensor_gpu.cpu()
            # Y_output_Array = Y_output_tensor_cpu.numpy()
            Y_output_Array = np.array(list(map(lambda x:x.cpu().numpy(), Y_output_tensor_gpu_list)))
            Y_output_Array = Y_output_Array.squeeze()

            savepath = os.path.join(self.TEST_RESULTS_FOLDER, f"model_V{self.READ_VERSION}_epoch_{CHECK_EPOCH}", f'{TEST_CASE_NAME}.npz')
            np.savez(savepath, input=Y_input_Array, output=Y_output_Array)
            #TEST_recorder_Dict[str(TEST_CASE_NAME)] = LOSS_ITEM
            #print(f'Model with [BP={self.CHECKBP}, Step={self.CHECKSTEP}, epoch={CHECK_EPOCH}] test loss in {TEST_CASE_NAME} : {average(LOSS_ITEM)}')

        return TEST_recorder_Dict


if __name__ == "__main__":
    print('*** Please run with \'TrainMethod.py\' or \'TestMethod.py\' script! ***')


                
