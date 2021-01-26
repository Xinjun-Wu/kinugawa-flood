import os
from tqdm import tqdm
import py7zr
import pandas as pd

area = ['Kinugawa','Kokaigawa']

for river in area:
    #E:\Wu\OneDrive - s.dlu.edu.cn\Kinugawa\CasesData
    input_folder  = f'E:\\Wu\\OneDrive - s.dlu.edu.cn\\{river}\\CasesData'
    output_folder = f'../Files Check Results/{river}'
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    name7z_List = os.listdir(input_folder)

    sheetbook = pd.DataFrame()
    for name7z in tqdm(name7z_List):
        path_7z = os.path.join(input_folder,name7z)
        BPNAME = name7z.split('.')[0]
        csvfile = f'{BPNAME}/case1/case1_1.csv'
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
            except Exception as e:
                print(f'    Error when decode  {name7z} :   {e}')

        #verify the Velocity (magnitude Max) of case1_1.csv
        csv_path = os.path.join(output_folder,csvfile)
        try:
            data = pd.read_csv(csv_path, header= 2)
            if max(data['Velocity (magnitude Max)']) > 0.0:
                caption = 'True'
            else:
                caption = '-'
        except Exception as e:
            print(f'    Error when read file  {csvfile} :   {e}')
            caption = 'Read Error'
        sheetbook.loc[f'{BPNAME}','case1_1 Velocity > 0.0'] = caption
    sheetbook.to_csv(f'../Check {river} Results.csv')

print('Done')

