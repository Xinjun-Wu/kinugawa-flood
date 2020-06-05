import pandas as pd 
import numpy as np 
import os 
from scipy import integrate
from tqdm import tqdm

class GenerateData():
    """ Class used for creating training npz files from case npy files.
    変換された水理解析結果NPYファイルを訓練用NPZファイルに作成する。
    作成時間間隔を指定する可能
    """
    def __init__(self, input_folder='../NpyData/', output_folder='../TrainData/', 
                timeinterval=10, n_delta=6, step=6, bpname='BP028'):
        self.INPUT_FOLDER = input_folder
        self.OUTPUT_FOLDER = output_folder
        self.TIMEINTERVAL = timeinterval
        self.N_DELTA = n_delta
        self.STEP = step
        self.BPNAME = bpname
        self.LOCATION = None
        self.N_TIMESTEMP = 0
        self.HEIGHT = 0
        self.ROWS = 0
        self.WIDTH = 0
        self.COLUMNS = 0
        self.NPZ_COUNT = 0
        if not os.path.exists(os.path.join(self.OUTPUT_FOLDER, f'Step_{self.STEP}')):
            os.makedirs(os.path.join(self.OUTPUT_FOLDER, f'Step_{self.STEP}'))

    def _walk_npy_folder(self):
        case_name_List = os.listdir(self.INPUT_FOLDER)
        case_name_List.remove('_info.npz')
        case_name_List.sort(key=lambda x:int(x.split('.')[0]))
        
        case_path_List = []
        for case_name in case_name_List:
            case_path_List.append(os.path.join(self.INPUT_FOLDER,case_name))

        example_data = np.load(case_path_List[0])
        self.N_TIMESTEMP = example_data.shape[0]
        self.HEIGHT = self.ROWS = example_data.shape[2]
        self.WIDTH = self.COLUMNS = example_data.shape[3]

        return case_name_List, case_path_List


    def _gengerate_sequence(self):
        raw_sequence = np.linspace(0, self.N_TIMESTEMP-1, self.N_TIMESTEMP, dtype=int)#[0,1,2,3,4,5,6,7,8,9,10,11,12,13]
        learning_sequence = raw_sequence[:-self.STEP]#[0,1,2,3,4,5,6,7]
        teacher_sequence = raw_sequence[self.STEP:]  #[6,7,8,9,10,11,12,13]

        integrate_sequence_DF = pd.DataFrame()
        for i in range(self.STEP+1):

            if (i-1) % self.N_DELTA == 0 :
                if i != 1:
                    integrate_sequence_DF[f"t{i-1}_"] = raw_sequence.copy()
                    integrate_sequence_DF[f"t{i-1}_"] = integrate_sequence_DF[f"t{i-1}_"].shift(-i+1)

            integrate_sequence_DF[f"t{i}"] = raw_sequence.copy()
            integrate_sequence_DF[f"t{i}"] = integrate_sequence_DF[f"t{i}"].shift(-i)
        
        integrate_sequence_DF = integrate_sequence_DF.dropna()
        integrate_sequence_DF = integrate_sequence_DF.astype(int)
        integrate_sequence = integrate_sequence_DF.to_numpy()
        return learning_sequence, teacher_sequence, integrate_sequence


    def _generate_data(self, case_name_List, case_path_List):
        learning_sequence, teacher_sequence, integrate_sequence = self._gengerate_sequence()
        BP_info = np.load(os.path.join(self.INPUT_FOLDER,'_info.npz'),allow_pickle=True)
        self.LOCATION = BP_info['IJPOINTS']
        Inflow_DF = BP_info['INFLOW_DF'][0]

        N_addchannel = int(self.STEP/self.N_DELTA)
        N_sample = self.N_TIMESTEMP-self.STEP
        inint_inflow_Array = np.zeros((N_sample, N_addchannel, self.HEIGHT, self.WIDTH))

        integrate_sequence = integrate_sequence.reshape(N_sample, N_addchannel, self.N_DELTA+1)
        
        for case_id, (case_name, case_path) in enumerate(tqdm(zip(case_name_List,case_path_List))):
            case_name_Str = f"case{case_name.split('.')[0]}"
            inflow_data = Inflow_DF[case_name_Str].to_numpy()
            watersituation = np.load(case_path)

            learning_value = watersituation[learning_sequence]
            teacher_value = watersituation[teacher_sequence]

            for sample_id in range(N_sample):
                integrate_index = integrate_sequence[sample_id]

                for channel_id in range(N_addchannel):
                    integrate_sub_index = integrate_index[channel_id]
                    integrate_item = inflow_data[integrate_sub_index]

                    result = integrate.simps(y = integrate_item.flatten(), dx = self.TIMEINTERVAL*60)

                    for location in self.LOCATION:

                        inint_inflow_Array[sample_id, channel_id, location[0]-1, location[1]-1 ] = result

            learning_data = np.concatenate((learning_value, inint_inflow_Array),axis = 1)
            teacher_data = teacher_value

            savename = os.path.join(self.OUTPUT_FOLDER, f'Step_{self.STEP}', case_name_Str[4:]+'.npz')
            np.savez(savename, learning_data=learning_data, teacher_data=teacher_data)
        self.NPZ_COUNT = case_id +1


    def run(self):
        case_name_List, case_path_List = self._walk_npy_folder()
        self._generate_data(case_name_List, case_path_List)
        print(f"Have generated {self.NPZ_COUNT} .npz files")


if __name__ == "__main__":

    BPNAME_List = ['BP120']
    TIMEINTERVAL = 10
    N_DELTA = 6
    STEP = 12

    for BPNAME in BPNAME_List:
        INPUT = f'../NpyData/{BPNAME}'
        OUTPUT = f'../TrainData/{BPNAME}'

        for h in range(6):
            STEP = 6*(h+1)
            print(f"\n Generating{BPNAME} {h+1}-hour data.")
            mygenerater = GenerateData(INPUT, OUTPUT, TIMEINTERVAL, N_DELTA, STEP, BPNAME)
            mygenerater.run()











            


