from genericpath import exists
import os
import numpy as np
import pandas as pd
from tqdm import tqdm


class Csv2Npy():

    def __init__(self, input_folder='../CasesData/BP028/', 
                        output_folder='../NpyData/BP028/',bpname='BP028'):
        self.INPUT_FOLDER = input_folder
        self.OUTPUT_FOLDER = output_folder
        self.NPY_COUNT = 0
        self.ROWS = None
        self.HEIGHT = None
        self.COLUMNS = None
        self.WIDTH = None
        self.BPNAME = bpname
        self.DEM = None

        # if not os.path.exists(self.OUTPUT_FOLDER):
        #     os.makedirs(self.OUTPUT_FOLDER)
    

    def _get_PointLists(self,listfile=r'破堤点格子番号.xlsx',skiprows=0,index_col=0):
        """
        获取破堤点网格番号
            
            Keyword Arguments:
                listfile {xlsx file} -- [argument for read file with pandas.read_excel] (default: {r'../氾濫流量ハイドロ/破堤点格子番号.xlsx'})
                skiprows {int} -- [argument for read file with pandas.read_excel] (default: {1})
                index_col {int} -- [argument for read file with pandas.read_excel] (default: {0})
            Returns:
                Points_I_J[list] -- [破堤点对应的 I J 坐标]
            Example:
                Points_I_J = [[270,1], [271,1], [272,1], [273,1], [274,1]]
        """
        pointlists = pd.read_excel(listfile, skiprows=skiprows, index_col=index_col)
        BPname = self.BPNAME
        # return a array : array[270,271,272,273,274.1]
        # points = pointlists.loc[BPname].to_numpy()[3:-1] 
        points = pointlists.loc[BPname].to_numpy()[3:9] 
        Points_I_J = []
        for i in range(len(points)-1):
            Points_I_J.append([int(points[i]),int(points[-1])])
        #Points_I_J = [[270,1], [271,1], [272,1, [273,1], [274,1]]
        Group_ID = pointlists.loc[BPname][0]
        return Group_ID, Points_I_J


    def _get_Inflow(self,inflowfile=r'氾濫ハイドロケース_10分間隔_20200127.xlsx',
                        header=0,sheet_name='氾濫ハイドロパターン (10分間隔)'):
        """
        获取各个工况下的入流记录
        
            Keyword Arguments:
                inflowfile {xlsxfile}} -- [description] (default: {r'../氾濫流量ハイドロ/氾濫ハイドロケース_10分間隔_20200127.xlsx'})
                header {int} -- [argument for read file with pandas.read_excel] (default: {0})
                sheet_name {str} -- [argument for read file with pandas.read_excel] (default: {'氾濫ハイドロパターン (10分間隔)'})
            
            Returns:
                inflow_DataFrame[pandas.DataFrame] -- [各工况的入流记录DataFrame纪录表，可用DataFrame['case1']提取数据]
        """
        inflow_DataFrame = pd.read_excel(inflowfile,header=header,sheet_name=sheet_name)
        return inflow_DataFrame


    def _get_info_(self):
        csvfile=self.INPUT_FOLDER+ os.sep + "case01/case01_1.csv"
        if not os.path.exists(csvfile):
            csvfile=self.INPUT_FOLDER+ os.sep + "case1/case1_1.csv"
        
        if not os.path.exists(csvfile):
                print(f'{csvfile}ファイルが既存していませんでした。\n')
        else:
            selectIJ=['I','J']
            selectXY=['X','Y']
            # I,J列タイプを指定
            dataframe = pd.read_csv(csvfile, header= 2,dtype={'I':np.int32,'J':np.int32})
            self.HEIGHT = self.ROWS = dataframe['I'].max() #get the rows of the target area
            self.WIDTH = self.COLUMNS = dataframe['J'].max() #get the columns of the target area
            #xyCoordinates.shape=(27030, 4) with I・J・X・Y
            ijCoordinates =dataframe[selectIJ].to_numpy()
            xyCoordinates =dataframe[selectXY].to_numpy()
            dem = dataframe['Elevation'].to_numpy()
            dem = dem.reshape(self.COLUMNS, self.ROWS).transpose()
            self.DEM = dem

        Group_ID, Points_I_J = self._get_PointLists()
        Inflow_DF = self._get_Inflow()

        npz_save_path = os.path.join(self.OUTPUT_FOLDER, 'Info')
        if not os.path.exists(npz_save_path):
            os.makedirs(npz_save_path)
        npz_save_name = os.path.join(npz_save_path,f'{self.BPNAME}_info.npz')
        np.savez(npz_save_name, IJCOORDINATES=ijCoordinates, XYCOORDINATES=xyCoordinates,
                HEIGHT=self.HEIGHT, WIDTH=self.WIDTH, IJPOINTS=Points_I_J, GROUP_ID=Group_ID, 
                INFLOW_DF=[Inflow_DF,0], DEM=[self.DEM, 0], allow_pickle=True)
        print(f'Have created _info.npz file for {self.BPNAME}.')
        return Group_ID
        

    def _get_data(self, casefolder_path):

        index_List = os.listdir(casefolder_path)
        # if except_list is not None:
        #     for except_file in except_list:
        #         index_List.remove(except_file)
        index_List.sort(key=lambda x:int(x.split('_')[1][:-4]))
        index_path_List = list(map(lambda x:os.path.join(casefolder_path,x), index_List))

        select=["Depth","Velocity(ms-1)X","Velocity(ms-1)Y"]
        watersituation = []

        for file_path in index_path_List: #循环n次, 这里n为时刻记数72
            data_PD = pd.read_csv(file_path, header= 2)

            #channel_i.shape=(27030, 3) with 水深・フラックスx・フラックy
            channel_i =data_PD[select].to_numpy()
            #(27030, 3)==>(53, 510, 3)==>(3,510, 53)
            channel_i=channel_i.reshape(self.COLUMNS,self.ROWS,3).transpose()
            #appending the data to list
            watersituation.append(channel_i)
        
        #(3,510, 53)==>(72, 3, 510, 53)
        watersituation = np.array(watersituation)

        return watersituation

    def _walk_cases(self, except_list=None):

        GROUP_ID = self._get_info_()
        casefolder_List = os.listdir(self.INPUT_FOLDER)
        if except_list is not None:
            for except_file in except_list:
                if except_file in casefolder_List:
                    casefolder_List.remove(except_file)
        casefolder_List.sort(key=lambda x:int(x.split('_')[0][4:]), reverse=False)

        #遍历每个case
        for casefolder in tqdm(casefolder_List):
            casefolder_path = os.path.join(self.INPUT_FOLDER,casefolder)
            casename_int = int(casefolder.split("_")[0][4:])
            watersituation = self._get_data(casefolder_path)
            #保存当前case的数据
            savepath = os.path.join(self.OUTPUT_FOLDER,f'{GROUP_ID}')
            if not os.path.exists(savepath):
                os.makedirs(savepath)
            savename = os.path.join(savepath,f'{self.BPNAME}_{casename_int:03}.npy')
            np.save(savename, watersituation)
            self.NPY_COUNT += 1
       

    def run(self,except_list):
        self._walk_cases(except_list)
        print(f"Have generated {self.NPY_COUNT} .npy files")

if __name__ == "__main__":
    BPNAME_List = ['BP028']
    BPNAME_List = [
            # 'BP019', 
            'BP022',
            'BP031',
            # 'BP016', 
            # 'BP025',
            # 'BP028',
            # 'BP037',
            # 'BP040',
            ]

    except_list = [
        # ['bp019.ipro'],
        ['bp022.ipro'],
        ['BP031.ipro'],
        # ['bp016.ipro'],
        # ['bp025.ipro'],
        # ['bp028.ipro'],
        # ['bp037.ipro'],
        # ['bp040.ipro'],
        ]

    for BPNAME,except_list_item in zip(BPNAME_List,except_list):

        INPUT = f'../CasesData/{BPNAME}'
        OUTPUT = f'../Save/alpha-cooperate Branch/NpyData'

        mynpy = Csv2Npy(INPUT,OUTPUT,BPNAME)
        mynpy.run(except_list_item)