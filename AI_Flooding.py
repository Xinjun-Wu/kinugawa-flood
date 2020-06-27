import numpy as np 
import pandas as pd
import os
import torch
from scipy import integrate
from Select_Net import select_net
import argparse
import sys

class AI_Model():
    def __init__(self, place_name, BP_ID, inflow_file, info_file, model_folder, save_path, device_type):
        self.PLACE_NAME = place_name
        self.BP_ID = BP_ID
        self.INFLOW_FILE = inflow_file
        self.INFO_FILE = info_file
        self.MODEL_FOLDER = model_folder
        self.SAVE_PATH = save_path
        
        self.N_DELTA = 6
        self.TIMEINTERVAL = 10
        self.N_TIMESTEMP = None
        self.STEP = None
        self.INFLOW_ARRAY = None
        self.INFLOW_POINTS = None

        self.GROUP_ID = None #the group id of name
        self.HEIGHT = None
        self.WIDTH = None
        self.LOCATION = None
        self.IJCOORDINATES = None
        self.XYCOORDINATES = None

        self.DEVICE_TYPE = device_type

        if self.DEVICE_TYPE == 'cuda' and torch.cuda.is_available():
            self.DEVICE = torch.device('cuda')
            print('Runing on the GPU...')
        else:
            self.DEVICE = torch.device('cpu')
            print('Runing on the CPU...')


        #check in
        if not os.path.exists(self.INFLOW_FILE):
            print(f'Cannot find the given folder or file at the following place: \r\n{ self.INFLOW_FILE}')
            sys.exit(0)
        if not os.path.exists(self.INFO_FILE):
            print(f'Cannot find the given folder or file at the following place: \r\n{ self.INFO_FILE }')
            sys.exit(0)

    def _loading(self):
        #loading inflow file(.csv)
        inflow_Array = pd.read_csv(self.INFLOW_FILE, header = None).to_numpy()
        self.INFLOW_ARRAY = inflow_Array
        self.N_TIMESTEMP = len(self.INFLOW_ARRAY)

        #loading informaton for current BP
        BP_info = np.load(self.INFO_FILE,allow_pickle=True)

        self.GROUP_ID = BP_info['Group_ID']
        self.HEIGHT = BP_info['HEIGHT']
        self.WIDTH = BP_info['WIDTH']
        self.LOCATION = BP_info['IJPOINTS']
        self.IJCOORDINATES = BP_info['IJCOORDINATES']
        self.XYCOORDINATES = BP_info['XYCOORDINATES']


        #loading model parameter
        hours = [1,2,3,4,5,6]
        steps = [6,12,18,24,30,36]
        models_parameter_path = os.listdir(self.MODEL_FOLDER)
        models_parameter_path.sort(key=lambda x:int(x.split('_')[1]))
        if len(models_parameter_path) != int(6):
            print(f'Cannot find the 6 predicting models at the following folder: \r\n{ self.MODEL_FOLDER }')
            sys.exit(0)

        models_parameter = []
        for param in models_parameter_path:
            if self.DEVICE_TYPE == 'cuda' and torch.cuda.is_available():
                models_parameter.append(torch.load(os.path.join(self.MODEL_FOLDER,param))['MODEL'])
            else:
                models_parameter.append(torch.load(os.path.join(self.MODEL_FOLDER,param),map_location='cpu')['MODEL'])
        self.MODELS_PARAMETER = models_parameter


    def _gengerate_sequence(self):
        raw_sequence = np.linspace(0, self.N_TIMESTEMP-1, self.N_TIMESTEMP, dtype=int)#[0,1,2,3,4,5,6,7,8,9,10,11,12,13]

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
        integrate_sequence = integrate_sequence_DF.to_numpy()[0]
        return integrate_sequence

    def _generate_data(self,step):

        self.STEP = step
        N_addchannel = int(step/self.N_DELTA)
        integrate_sequence = self._gengerate_sequence()
        inint_inflow_Array = np.zeros((1, N_addchannel, self.HEIGHT, self.WIDTH))
        integrate_sequence = integrate_sequence.reshape(1, N_addchannel, self.N_DELTA+1)
        
        inflow_data = self.INFLOW_ARRAY
        watersituation = np.zeros((1,3,self.HEIGHT,self.WIDTH))
        watersituation[0,0] = 0.01

        integrate_index = integrate_sequence[0]

        for channel_id in range(N_addchannel):
            integrate_sub_index = integrate_index[channel_id]
            integrate_item = inflow_data[integrate_sub_index]
            result = integrate.simps(y = integrate_item.flatten(), dx = self.TIMEINTERVAL*60)
            
            for location in self.LOCATION:
                inint_inflow_Array[0, channel_id, location[0]-1, location[1]-1 ] = result

            input_data = np.concatenate((watersituation, inint_inflow_Array),axis = 1)

        return input_data

    def _prediction(self,model,input_data):
        model.eval()

        with torch.no_grad():
            input_tensor = torch.tensor(input_data,device=self.DEVICE,dtype=torch.float32)

            model.zero_grad()
            output_tensor = model(input_tensor)
            output_tensor_cpu = output_tensor.cpu()
            output_Array = output_tensor_cpu.numpy()
            output_Array = output_Array.reshape(3,self.HEIGHT,self.WIDTH)

        return output_Array

    def run(self):
        
        self._loading()
        # print(f"Loading parameters of {self.PLACE_NAME} {self.BP_ID}...")
        # # 1-6 hour
        # print('Predicting...')

        output_dic = {
                        'I':self.IJCOORDINATES[:,0],
                        'J':self.IJCOORDINATES[:,1],
                        'X':self.XYCOORDINATES[:,0],
                        'Y':self.XYCOORDINATES[:,1],
                        }

        #for h in tqdm(range(1,7)):
        for h in range(1,7):
            step = h*6
            input_data = self._generate_data(step)

            model = select_net(self.GROUP_ID,h+3)
            model.to(self.DEVICE)
            param = self.MODELS_PARAMETER[h-1]

            model.load_state_dict(param)

            output_Array = self._prediction(model, input_data)

            depth = output_Array[0].transpose()
            velocity_x = output_Array[1].transpose()
            velocity_y = output_Array[2].transpose()
            
            output_dic[f"Depth-{h}hour"] = depth.ravel()
            output_dic[f"Velocity(ms-1)X-{h}hour"] = velocity_x.ravel()
            output_dic[f"Velocity(ms-1)Y-{h}hour"] = velocity_y.ravel()
            

        if not os.path.exists(self.SAVE_PATH):
            os.makedirs(self.SAVE_PATH)
        savename = os.path.join(self.SAVE_PATH,f"Output.csv")
        output_pd = pd.DataFrame(output_dic)
        output_pd.to_csv(savename,index=False,float_format='%.6f')
        print('Done!\r\n')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('PLACE_NAME')
    parser.add_argument("BP_ID")
    parser.add_argument(('DEVICE_TYPE'))
    args = parser.parse_args()
    parser.parse_args()

    PLACE_NAME = args.PLACE_NAME
    BP_ID = args.BP_ID
    DEVICE_TYPE = args.DEVICE_TYPE

    # PLACE_NAME = 'Kokaigawa'
    # BP_ID = 'BP120'
    # DEVICE_TYPE = 'cpu'

    INFLOW_FILE = f'../{PLACE_NAME}/{BP_ID}/Input/inflow.csv'
    POINT_FILE = f'../{PLACE_NAME}/{BP_ID}/Information/_info.npz'
    MODEL_FOLDER = f'../{PLACE_NAME}/{BP_ID}/Model/'
    SAVE_PATH = f'../{PLACE_NAME}/{BP_ID}/Output/'
    
    my_AI = AI_Model(PLACE_NAME,BP_ID,INFLOW_FILE,POINT_FILE,MODEL_FOLDER,SAVE_PATH,DEVICE_TYPE)
    my_AI.run()

    


            



                
            


