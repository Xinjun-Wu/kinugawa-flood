import numpy as np 
import torch
import torch.utils.data as Data
import os
import random
import time
import datetime

class KinugawaDataSets():
    def __init__(self, step, input_folder='../TrainData/BP028/Step_6/',
                tvt_ratio=[0.5,0.3,0.2], test_specific=[10, 11], random_seed=120,bpname='BP028'):
        """
        数据将按照tvt_ratio的比例划分train,validdaton,test数据集,指定的test_specific必定在测试集内

        """
        self.STEP = step
        self.BPNAME = bpname
        self.INPUT_FOLDER = input_folder
        self.TVT_RATIO = tvt_ratio
        self.TEST_SPECIFIC = test_specific
        self.RANDOM_SEED = random_seed
        self.N_CASE = 0
        self.N_TRAIN = 0
        self.N_VALIDATION = 0
        self.N_TEST= 0
        self.TRAIN_LIST = []
        self.VALIDATION_LIST = []
        self.TEST_LIST = []
        self.N_SAMPLE_EACHCASE = 0
        self.N_CHANNEL = 0
        self.HEIGHT = 0
        self.ROWS = 0
        self.WIDTH = 0
        self.COLUMNS = 0
        self._assign_case()


    def _assign_case(self):
        """
        分配数据集
        """
        casename_List = os.listdir(self.INPUT_FOLDER)

        self.N_CASE = len(casename_List)
        N_train = int(self.N_CASE * self.TVT_RATIO[0])
        N_validation = int(self.N_CASE * self.TVT_RATIO[1])
        N_test = int(self.N_CASE * self.TVT_RATIO[2])

        random.Random(self.RANDOM_SEED).shuffle(casename_List)

        self.TRAIN_LIST = casename_List[:N_train]
        self.VALIDATION_LIST = casename_List[N_train:N_train+N_validation]
        self.TEST_LIST = casename_List[-N_test:]

        for s in self.TEST_SPECIFIC:
            item = f"{int(s)}.npz"
            if item not in self.TEST_LIST:
                self.TEST_LIST.append(item)
                N_test += 1

                if item in self.TRAIN_LIST:
                    self.TRAIN_LIST.remove(item)
                    N_train -= 1

                elif item in self.VALIDATION_LIST:
                    self.VALIDATION_LIST.remove(item)
                    N_validation -= 1
        
        self.N_TRAIN = N_train
        self.N_VALIDATION = N_validation
        self.N_TEST = N_test

        self.TRAIN_LIST.sort(key=lambda x:int(x.split(".")[0]))
        self.VALIDATION_LIST.sort(key=lambda x:int(x.split(".")[0]))
        self.TEST_LIST.sort(key=lambda x:int(x.split(".")[0]))

        example_data = np.load(os.path.join(self.INPUT_FOLDER, self.TRAIN_LIST[0]))
        learning_data = example_data['learning_data']
        #teacher_data = example_data['teacher_data']
        self.N_SAMPLE_EACHCASE = learning_data.shape[0]
        self.N_CHANNEL = learning_data.shape[1]
        self.HEIGHT = self.ROWS = learning_data.shape[2]
        self.WIDTH = self.COLUMNS =  learning_data.shape[3]


        print(f'指定路径内一共有{self.N_CASE}个case,按照给定参数划分如下：')
        print(f"训练集{self.N_TRAIN}，验证集{self.N_VALIDATION}，测试集{self.N_TEST}。")
        test_print = [int(x.split('.')[0]) for x in self.TEST_LIST]
        print(f'测试集的case为：{test_print}')

    
    def select(self, data='train', dataset_type='tensorstyle'):

        if data == 'train':
            selected_id = 0
        elif data == 'validation':
            selected_id = 1
        elif data == 'test':
            selected_id = 2

        index_container = [self.TRAIN_LIST, self.VALIDATION_LIST, self.TEST_LIST]
        case_num_container = [self.N_TRAIN, self.N_VALIDATION, self.N_TEST]

        selected_List = index_container[selected_id]
        selected_num = case_num_container[selected_id]

        case_path_List = []#存放case的可访问路径
        case_index_List = []#存放case的序号名，比如1，2，
        dataset_info = {}#存放选择的dataset的基本信息

        for case in selected_List:
            case_path_List.append(os.path.join(self.INPUT_FOLDER, case))
            case_index_List.append(int(case.split(".")[0]))
        
        N_sample_total = self.N_SAMPLE_EACHCASE * selected_num
        dataset_info = {
                        'n_case' : selected_num,
                        'n_sample_eachcase' : self.N_SAMPLE_EACHCASE,
                        'n_sample_total' : N_sample_total,
                        'n_channel' : self.N_CHANNEL,
                        'height' : self.HEIGHT,
                        'width' : self.WIDTH,
                        'rows' : self.ROWS,
                        'columns' : self.COLUMNS,
                        'case_index_List' : case_index_List,
                        'case_path_List' : case_path_List
                        }
        if dataset_type == 'mapstyle':
            dataset = Map_style_DataSet(dataset_info)
            
        elif dataset_type == 'tensorstyle':
            X_Array_List = []
            y_Array_List = []
            for case_path in case_path_List:
                case_data = np.load(case_path)            
                learning_data = case_data['learning_data']
                teacher_data = case_data['teacher_data']
                X_Array_List.append(learning_data)
                y_Array_List.append(teacher_data)

            X_Array = np.concatenate(tuple(X_Array_List), axis=0)
            y_Array = np.concatenate(tuple(y_Array_List), axis=0)

            X_Tensor = torch.tensor(X_Array)
            y_Tensor = torch.tensor(y_Array)
            dataset = Data.TensorDataset(X_Tensor,y_Tensor)

        return dataset_info, dataset




class Map_style_DataSet(Data.Dataset):
    def __init__(self,info_Dict):
        super(Map_style_DataSet).__init__()
        self.N_CASE = info_Dict['n_case']
        self.N_SAMPLE_EACHCASE = info_Dict['n_sample_eachcase']
        self.N_SAMPLE_TOTAL = info_Dict['n_sample_total']
        self.CASE_INDEX_LIST = info_Dict['case_index_List']
        self.CASE_PATH_LIST = info_Dict['case_path_List']
        self.TEMP_DATA = None
        self.TEMP_CASE_ID = None


    def __getitem__(self, index):
        case_id = index // self.N_SAMPLE_EACHCASE
        sample_id = index % self.N_SAMPLE_EACHCASE

        if case_id == self.TEMP_CASE_ID:
            case_data = self.TEMP_DATA
        else :
            case_data = np.load(self.CASE_PATH_LIST[case_id])
            self.TEMP_CASE_ID = case_id
            self.TEMP_DATA = case_data
        
        learning_data = case_data['learning_data']
        teacher_data = case_data['teacher_data']

        X_Array = learning_data[sample_id]
        y_Array = teacher_data[sample_id]

        return X_Array, y_Array

    def __len__(self):
        return self.N_SAMPLE_TOTAL
    

# class Tensor_DataSet(Data.TensorDataset):
#     def __init__(self, info_Dict):
#         super(Tensor_DataSet).__init__()
#         self.N_CASE = info_Dict['n_case']
#         self.N_SAMPLE_EACHCASE = info_Dict['n_sample_eachcase']
#         self.N_SAMPLE_TOTAL = info_Dict['n_sample_total']
#         self.CASE_INDEX_LIST = info_Dict['case_index_List']
#         self.CASE_PATH_LIST = info_Dict['case_path_List']

#         X_Array_List = []
#         y_Array_List = []
#         for case_path in self.CASE_PATH_LIST:
#             case_data = np.load(case_path)            
#             learning_data = case_data['learning_data']
#             teacher_data = case_data['teacher_data']
#             X_Array_List.append(learning_data)
#             y_Array_List.append(teacher_data)

#         X_Array = np.concatenate(tuple(X_Array_List), axis=0)
#         y_Array = np.concatenate(tuple(y_Array_List), axis=0)

#         X_Tensor = torch.tensor(X_Array)
#         y_Tensor = torch.tensor(y_Array)

#         self.tensors = (X_Tensor, y_Tensor)
#     def __getitem__(self, index):
#         return tuple(tensor[index] for tensor in self.tensors)

#     def __len__(self):
#         return int(self.N_CASE) * int(self.N_SAMPLE_EACHCASE)


        
if __name__ == "__main__":
    mydataset = KinugawaDataSets(6)
    data_info, trainsets = mydataset.select('train')
    traindataloder = Data.DataLoader(dataset=trainsets, batch_size=10, shuffle=True, num_workers = 3,pin_memory=True)

    start_clock = time.time()
    start_total = start_clock
    for batch_id, (X_Tensor, y_Tensor) in enumerate(traindataloder):

        end_clock = time.time()
        start = datetime.timedelta(seconds=start_clock)
        end = datetime.timedelta(seconds=end_clock)
        timeusage = str(end - start)

        print(f"batch_id: {batch_id}, timeusage: {timeusage}, type: {type(X_Tensor)}, size: {X_Tensor.size()}")
        start_clock = time.time()
    end_total = time.time()
    timeusage = str(datetime.timedelta(seconds=end_total) - datetime.timedelta(seconds=start_total))
    print(f'timeusage: {timeusage}')
    



