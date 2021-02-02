
from random import shuffle
import numpy as np 
import torch
import torch.utils.data as Data
import os
import random
import time
import datetime
from sklearn.model_selection import train_test_split

class CustomizeDataSets():
    def __init__(self, input_folder='../TrainData',step = 1, group_id = 'Ki1', 
                except_bp = ['BP032'],only_bp = ['BP028'], except_case = ['BP028_001, BP032_014'], 
                test_size=0.3, shuffle = True, random_seed=120):
        """
        数据将按照tvt_ratio的比例划分train,validdaton,test数据集

        """
        self.INPUT_FOLDER = input_folder
        self.STEP = step
        self.GROUP_ID = group_id
        self.EXCEPT_BP = except_bp
        self.ONLY_BP = only_bp
        self.EXCEOT_CASE = except_case
        self.TEST_SIZE = test_size
        self.RANDOM_SEED = random_seed
        self.SHUFFLE = shuffle
        self.N_SAMPLE = 0
        self.N_CHANNEL = 0
        self.HEIGHT = 0
        self.ROWS = 0
        self.WIDTH = 0
        self.COLUMNS = 0
        self.TEMP_DATA = None


    def _walk_npz_folder(self):
        """read the name and the path of each case with return two list

        Returns:
            case_name_List [list]: 
            case_path_List [list]:
        """
        files_List = os.listdir(os.path.join(self.INPUT_FOLDER,f"Step_{int(self.STEP):02}",self.GROUP_ID))
        files_List_bk = files_List.copy()
        if self.ONLY_BP is not None:
            for bp in self.ONLY_BP:
                i = 0
                for file in files_List_bk:
                    if file.split('_')[0] != bp:
                        files_List.remove(file)
                    else:
                        i += 1 #count the num of ONLP_BP
                if i == 0:
                    raise ValueError(f'No {bp} in the target folder!')
        else:
            if self.EXCEPT_BP is not None:
                for bp in self.EXCEPT_BP:
                    i = 0
                    for file in files_List_bk:
                        if file.split('_')[0] == bp:
                            files_List.remove(file)
                            i += 1
                    if i == 0:
                        raise ValueError(f'No {bp} in the target folder!')
        
        files_List_bk = files_List.copy()
        if self.EXCEOT_CASE is not None:
            for case in self.EXCEOT_CASE:
                j = 0
                for file in files_List_bk:
                    if file.split('.')[0] == case:
                        files_List.remove(file)
                        j += 1
                if j == 0:
                    raise ValueError(f'No {case} in the target folder!')

        case_name_List = files_List.copy()
        case_name_List.sort(key=lambda x:int(x.split('_')[0][2:])) # BP028_001 ==> 028 ==> 28 
        
        case_path_List = []
        for case_name in case_name_List:
            case_path_List.append(os.path.join(self.INPUT_FOLDER,f"Step_{int(self.STEP):02}",self.GROUP_ID,case_name))

        example_data = np.load(case_path_List[0]) # 读取第一个case的数据
        self.N_SAMPLE = example_data['learning_data'].shape[0] # 返回case中样本数量
        self.N_CHANNEL = example_data['learning_data'].shape[1] # 返回case的通道数
        self.HEIGHT = self.ROWS = example_data['learning_data'].shape[2] #返回case网格的高度个数，例如510
        self.WIDTH = self.COLUMNS = example_data['learning_data'].shape[3]#返回case网格宽度个数，例如53

        return case_name_List, case_path_List


    def _load_data(self):
        #加载数据
        X_list = []
        y_list = []

        case_name_List, case_path_List = self._walk_npz_folder()
        #遍历所有case
        for case_id, (case_name, case_path) in enumerate(zip(case_name_List, case_path_List)):
            #从文件读取数据
            case_data = np.load(case_path)
            learning_data = case_data["learning_data"] #返回的是四阶数组
            teacher_data = case_data['teacher_data']

            X_list.append(learning_data)
            y_list.append(teacher_data)

        X_array = np.array(X_list).reshape(-1, self.N_CHANNEL, self.HEIGHT, self.WIDTH)#通道数不确定
        y_array = np.array(y_list).reshape(-1, 3, self.HEIGHT, self.WIDTH)#通道数为3
        X_train, X_test, y_train, y_test = train_test_split(X_array, y_array, test_size=0.30, random_state=self.RANDOM_SEED)
        return  X_train,X_test,y_train,y_test

    def select(self, data='train', dataset_type='tensorstyle'):
        #加载数据并临时存在内存中
        if self.TEMP_DATA == None:
            X_train,X_test,y_train,y_test = self._load_data()
            self.TEMP_DATA = [X_train,X_test,y_train,y_test]
        else:
            X_train = self.TEMP_DATA[0]
            X_test = self.TEMP_DATA[1]
            y_train = self.TEMP_DATA[2]
            y_test = self.TEMP_DATA[3]

        if data == 'train':

            X_tensor = torch.tensor(X_train)
            y_tensor = torch.tensor(y_train)

        elif data == 'test':

            X_tensor = torch.tensor(X_test)
            y_tensor = torch.tensor(y_test)

        dataset = Data.TensorDataset(X_tensor, y_tensor)

        return dataset

        
if __name__ == "__main__":

    INPUT_FOLDER = f'../TrainData'
    STEP = 1
    GROUP_ID = 'Ki1'
    EXCEPT_BP = None
    ONLP_BP = ["BP028"]
    EXCEPT_CASE = ['BP028_001','BP028_014']
    
    TEST_SIZE = 0.3
    SHUFFLE = True
    RANDOM_SEED=120

    mydataset = CustomizeDataSets(INPUT_FOLDER,STEP,GROUP_ID,EXCEPT_BP,ONLP_BP, EXCEPT_CASE,
                                    TEST_SIZE,SHUFFLE,RANDOM_SEED)
    trainsets = mydataset.select('train')
    traindataloder = Data.DataLoader(dataset=trainsets, batch_size=100, shuffle=True, num_workers = 3,
                                    pin_memory=True,drop_last=True)
    # 开始
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
    



