import pandas as pd 
import numpy as np 
import os 
from scipy import integrate
from sklearn import preprocessing
from tools import area_extract
import argparse
import time

class GenerateData():
    """ Class used for creating training npz files from case npy files.
    変換された水理解析結果NPYファイルを訓練用NPZファイルに作成する。
    作成時間間隔を指定する可能
    """
    def __init__(self, input_folder='../NpyData', output_folder='../TrainData', 
                group_id = 'Ki1' ,bpname='BP028',
                timeinterval=10, n_delta=6, step=6, ):
        self.INPUT_FOLDER = input_folder
        self.OUTPUT_FOLDER = output_folder
        self.TIMEINTERVAL = timeinterval
        self.N_DELTA = n_delta
        self.STEP = step
        self.GROUP_ID = group_id
        self.BPNAME = bpname
        self.LOCATION = None
        self.N_TIMESTEMP = 0
        self.HEIGHT = 0
        self.ROWS = 0
        self.WIDTH = 0
        self.COLUMNS = 0
        self.NPZ_COUNT = 0
        self.DEM = None
        if not os.path.exists(os.path.join(self.OUTPUT_FOLDER, f'Step_{int(self.STEP):02}',f"{self.GROUP_ID}")):
            os.makedirs(os.path.join(self.OUTPUT_FOLDER, f'Step_{int(self.STEP):02}',f"{self.GROUP_ID}"))

    def _walk_npy_folder(self):
        """read the name and the path of each case with return two list

        Returns:
            case_name_List [list]: 
            case_path_List [list]:
        """
        files_List = os.listdir(os.path.join(self.INPUT_FOLDER, self.GROUP_ID))
        case_name_List = []
        for file in files_List:
            if file.split('_')[0] == self.BPNAME:
                case_name_List.append(file)
        #case_name_List.remove('_info.npz')
        case_name_List.sort(key=lambda x:int(x.split('.')[0][-3:]))
        
        case_path_List = []
        for case_name in case_name_List:
            case_path_List.append(os.path.join(self.INPUT_FOLDER, self.GROUP_ID, case_name))

        example_data = np.load(case_path_List[0]) # 读取第一个case的数据
        self.N_TIMESTEMP = example_data.shape[0] # 返回case中样本数量，例如72个时刻，会包含72个不同时刻的样本
        self.HEIGHT = self.ROWS = example_data.shape[2] #返回case网格的高度个数，例如510
        self.WIDTH = self.COLUMNS = example_data.shape[3]#返回case网格宽度个数，例如53

        return case_name_List, case_path_List


    def _gengerate_sequence(self):
        """返回学习值，教师值和积分值的index

        """
        #构建总步长进行编号，赋予等差从0增加的步长为1的整数数值index
        raw_sequence = np.linspace(0, self.N_TIMESTEMP-1, self.N_TIMESTEMP, dtype=int)#[0,1,2,3,4,5,6,7,8,9,10,11,12,13]
        #构建学习值序列和教师值序列进行步长编号
        learning_sequence = raw_sequence[:-self.STEP]#[0,1,2,3,4,5,6,7]
        teacher_sequence = raw_sequence[self.STEP:]  #[6,7,8,9,10,11,12,13]

        integrate_sequence_DF = pd.DataFrame()
        for i in range(self.STEP+1):
            #判断前一个步长编号能否北N_DELTA整除，如果能，则将前一个步长的值进行复制并排列在其后面
            #如果不能，则将本步长对应的值排列在其后面
            if (i-1) % self.N_DELTA == 0 :
                if i != 1 and i != 0: # i 为0和1时 为特殊情况，此时不需要对前一个步长进行复制，
                                    #因为前面没有既有序列或者既有序列无需重复
                    integrate_sequence_DF[f"t{i-1}_"] = raw_sequence.copy()
                    integrate_sequence_DF[f"t{i-1}_"] = integrate_sequence_DF[f"t{i-1}_"].shift(-i+1)

            integrate_sequence_DF[f"t{i}"] = raw_sequence.copy()
            integrate_sequence_DF[f"t{i}"] = integrate_sequence_DF[f"t{i}"].shift(-i)
        #剔除na的值，保留有有效序列的行
        integrate_sequence_DF = integrate_sequence_DF.dropna()
        integrate_sequence_DF = integrate_sequence_DF.astype(int)
        integrate_sequence = integrate_sequence_DF.to_numpy()
        return learning_sequence, teacher_sequence, integrate_sequence


    def _generate_data(self, case_name_List, case_path_List):
        """generating data for each case

        Args:
            case_name_List ([list]): [description]
            case_path_List ([list]): [description]
        """
        #生成学生值，教师值，和入流积分的序列
        learning_sequence, teacher_sequence, integrate_sequence = self._gengerate_sequence()
        #提取当前case的信息，比如高宽，破堤点，DEM
        BP_info = np.load(os.path.join(self.INPUT_FOLDER,'Info',f'{self.BPNAME}_info.npz'),allow_pickle=True)
        self.LOCATION = BP_info['IJPOINTS'] # 提取破地点，二级list [[I,J],[I,J]] I为高度即行数，J为宽度即列数
        Inflow_DF = BP_info['INFLOW_DF'][0] # 提取破地点的所有case的入流情况，以DataFrame形式返回
        DEM = BP_info['DEM'][0] #提取当前研究区域的绝对数字高程数据，以ndarray形式返回

        #对DEM进行归一化处理，采用MinMaxScaler
        scaler = preprocessing.MinMaxScaler()
        DEM_MinMax = scaler.fit_transform(DEM)
        #转化为array并拓展为四阶Array,并于后面broadcast
        DEM_MinMax_Array = np.array(DEM_MinMax)
        DEM_MinMax_Array = np.tile(DEM_MinMax_Array,(self.N_TIMESTEMP-self.STEP,1,1,1))
        
        N_addchannel = int(self.STEP/self.N_DELTA) # 根据计算步长 %STEP% 及分段积分步长 %N_DELTA% 计算增加的通道个数
        N_sample = self.N_TIMESTEMP-self.STEP # 根据时刻计数即总步长 %N_TIMESTEMP% 和计算步长 %STEP% 计算完整的样本个数
        inint_inflow_Array = np.zeros((N_sample, N_addchannel, self.HEIGHT, self.WIDTH))#初始化,

        integrate_sequence = integrate_sequence.reshape(N_sample, N_addchannel, self.N_DELTA+1)
        
        # for case_id, (case_name, case_path) in enumerate(tqdm(zip(case_name_List,case_path_List))):
        for case_name, case_path in zip(case_name_List,case_path_List):
            case_name_Str = f"case{int(case_name.split('.')[0][-3:])}" # BP028_001 ==> case1
            inflow_data = Inflow_DF[case_name_Str].to_numpy() #当前case的入流数据
            watersituation = np.load(case_path) # 当前case的水流状态 numpy的shape 为(N, C, H, W)
            #提取按照时刻对应的学生值和教师值的水流状态 numpy的shape 为(N_sample, C, H, W)
            learning_value = watersituation[learning_sequence] 
            teacher_value = watersituation[teacher_sequence]

            #对某个case内做sample的循环，每个case包含N_sample个样本
            for sample_id in range(N_sample):

                #对教师值和学生值的范围进行掩码处理，非研究区域的值归零

                learning_sample = learning_value[sample_id]
                learning_sample[0] = area_extract(learning_sample[1],learning_sample[0],0,0,0,None) #通过mask根据水流速度范围提取水深的范围
                learning_value[sample_id] = learning_sample

                teacher_sample = teacher_value[sample_id]
                teacher_sample[0] = area_extract(teacher_sample[1],teacher_sample[0],0,0,0,None) #通过mask根据水流速度范围提取水深的范围
                teacher_value[sample_id] = teacher_sample


                #提取当前sample需要积分的入流index,以二级list[[0,1,2],[2,3,4]]的形式返回
                integrate_index = integrate_sequence[sample_id] 

                for channel_id in range(N_addchannel):
                    #根据不同的channel_id提取不同的入流index, 以list [0,1,2]的形式返回
                    integrate_sub_index = integrate_index[channel_id]
                    #根据入流index 提取对应的入流数值
                    integrate_item = inflow_data[integrate_sub_index]
                    #入流积分进行辛普森公式积分
                    result = integrate.simps(y = integrate_item.flatten(), dx = self.TIMEINTERVAL*60)
                    result = result/10000000 #进行小数定标标准化

                    #积分结果填充到初始入流数组的对应位置
                    for location in self.LOCATION:
                        inint_inflow_Array[sample_id, channel_id, location[0]-1, location[1]-1 ] = result
            
            #在第一阶上连接学习值和入流值
            # learning_data = np.concatenate((DEM_MinMax_Array, learning_value, inint_inflow_Array),axis = 1)
            learning_data = np.concatenate((DEM_MinMax_Array, learning_value, inint_inflow_Array),axis = 1)
            teacher_data = teacher_value

            print(f'Saving {self.BPNAME},{case_name_Str}')
            # for i in tqdm(range(learning_data.shape[0])):
            #     savename = os.path.join(self.OUTPUT_FOLDER, f'Step_{int(self.STEP):02}',f"{self.GROUP_ID}",
            #                             f"{self.BPNAME}",
            #                             f"{self.BPNAME}_{int(case_name_Str[4:]):03}_{i:03}.npz")
            #     np.savez(savename, learning_data=learning_data[i], teacher_data=teacher_data[i])
            #     #print(f'Saving {savename}')
            savename = os.path.join(self.OUTPUT_FOLDER, f'Step_{int(self.STEP):02}',f"{self.GROUP_ID}",
                                    f"{self.BPNAME}_{int(case_name_Str[4:]):03}.npz")
            np.savez(savename, learning_data=learning_data, teacher_data=teacher_data)

        #self.NPZ_COUNT = len(case_name_List) +1


    def run(self):
        case_name_List, case_path_List = self._walk_npy_folder()
        self._generate_data(case_name_List, case_path_List)
        #print(f"Have generated {self.NPZ_COUNT} .npz files")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    #parser.add_argument('GROUP')
    parser.add_argument('BPNAME')
    args = parser.parse_args()

    #GROUP_ID = args.GROUP
    BPNAME = args.BPNAME

    print(f'{BPNAME}  generate data start: {time.ctime()}\r\n')

    BRANCH = 'Master Branch'
    # BRANCH = 'Academic Branch'
    # BRANCH = 'Cooperate Branch'
    # BRANCH = 'Dev Branch'

    TIMEINTERVAL = 10
    N_DELTA = 1
    STEP = 1

    INPUT = f'../Save/{BRANCH}/NpyData'
    OUTPUT = f'../Save/{BRANCH}/TrainData'

    #读取信息描述文件，提取破堤区域数值模拟网格代号GROUP_ID
    INFO_path = f'../Save/{BRANCH}/NpyData/Info/{BPNAME[:5]}_info.npz'
    INFO_file = np.load(INFO_path)
    GROUP_ID = INFO_file['GROUP_ID']
    
    print(f"Generating {BPNAME} STEP={STEP} data.")
    mygenerater = GenerateData(INPUT, OUTPUT, GROUP_ID, BPNAME,TIMEINTERVAL, N_DELTA, STEP, )
    mygenerater.run()

    print(f'\r\n{BPNAME} generate data end: {time.ctime()}')











            


