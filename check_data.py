import os
from tqdm import tqdm
import py7zr
import pandas as pd
import re
import shutil

area = ['Kinugawa','Kokaigawa']

for river in area:
    #E:\Wu\OneDrive - s.dlu.edu.cn\Kinugawa\CasesData
    # input_folder  = f'E:\\Wu\\OneDrive - s.dlu.edu.cn\\{river}\\CasesData'
    input_folder  = f'C:\\Users\\bent2\\OneDrive - s.dlu.edu.cn\\{river}\\CasesData'
    output_folder = f'../Files Check Results/{river}'
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    name7z_List = os.listdir(input_folder)
    
    filter = re.compile('^BP\d\d\d.7z$',flags=0)
    name7z_List_filted = [f for f in name7z_List if filter.match(f) ]

    name7z_List = name7z_List_filted

    sheetbook = pd.DataFrame()
    for name7z in tqdm(name7z_List):
        path_7z = os.path.join(input_folder,name7z)
        BPNAME = name7z.split('.')[0]
        csvfile = f'{BPNAME}/case1/case1_1.csv'
        csvfile_ = f'{BPNAME}/case01/case01_1.csv'
        with py7zr.SevenZipFile(path_7z, 'r') as archive:
            # verify the CRCc value of the compressed files 
            try :
                CRCS_value = archive.testzip()
                if CRCS_value is not None:
                    sheetbook.loc[f'{BPNAME}','Decode Error'] = 'True'
                else:
                    sheetbook.loc[f'{BPNAME}','Decode Error'] = '-'
            except Exception as e:
                sheetbook.loc[f'{BPNAME}','Decode Error'] = 'Unknow'
                print(f'    Error when verify the {BPNAME} compressed file :   {e}')
            
        with py7zr.SevenZipFile(path_7z, 'r') as archive:
            #decode the case1_1
            try:
                archive.extract(output_folder,csvfile)
                archive.extract(output_folder,csvfile_)
            except Exception as e:
                print(f'    Error when decode  {name7z} :   {e}')

        #verify the Velocity (magnitude Max) of case1_1.csv
        csv_path = os.path.join(output_folder,csvfile)
        csv_path_ = os.path.join(output_folder,csvfile_)

        data = 0 #初始化数据为0
        caption = '-'

        # read csv
        try:
            data = pd.read_csv(csv_path, header= 2)
        except FileNotFoundError:
            try:
                data = pd.read_csv(csv_path_, header= 2)
            except Exception as e:
                print(f'    Error when read file  {csvfile} :   {e}')
                caption = 'Read Error'
        except Exception as e:
            print(f'    Error when read file  {csvfile} :   {e}')
            caption = 'Read Error'


        if caption is not 'Read Error':
            if max(data['Velocity (magnitude Max)']) > 0.0:
                caption = 'True'
            else:
                caption = '-'
            sheetbook.loc[f'{BPNAME}','case1_1 Velocity > 0.0'] = caption
        BPpath = os.path.join(output_folder,BPNAME)
        shutil.rmtree(BPpath)
        print(f'Remove {BPpath}')

    sheetbook.to_csv(f'../Files Check Results/Check {river} Results.csv')

    shutil.rmtree(output_folder)
    print(f'Remove {output_folder}')

print('Done')

