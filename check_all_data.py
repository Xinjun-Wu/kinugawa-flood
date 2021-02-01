import os
from tqdm import tqdm
import py7zr
import pandas as pd
import re
import shutil

def extract_7z(path_7z,name_7z,extract_path,verify = True, overwrite = False):
    CRCS_value = 0
    caption = 'OK'

    # check the exists of extract path of current BP
    if not os.path.exists(os.path.join(extract_path,name_7z.split('.')[0])) or overwrite:

        if verify:
            print(f'Verifying {name_7z} ...')
            with py7zr.SevenZipFile(path_7z, 'r') as archive:
                # verify the CRCc value of the compressed files 
                try :
                    CRCS_value = archive.testzip()
                except Exception as e:
                    print(f'    Error when verify the {name_7z}:   {e}')
                    caption = 'Extract Error'
        else:
            CRCS_value = None

        if CRCS_value is None:

                    print(f'Extracting {name_7z} ...')
                    with py7zr.SevenZipFile(path_7z, 'r') as archive:
                        try:
                            archive.extractall(extract_path)
                            print(f'Extracte {name_7z} succeddfully!')
                        except Exception as e:
                            print(f'    Error when extract the {name_7z}:   {e}')
                            caption = 'Extract Error'
    return caption

def check_bpcase(mainsheetbook,bpname,bppath,csv_rows,savepath,nums=31,csv_overwrite=False):

    bpcase_caption = 'OK'
    # read and filter the name of case ,such as 'case1' or 'case01'
    dir_names = os.listdir(bppath)

    filter1 = re.compile('^case\d$',flags=0)
    filter2 = re.compile('^case\d\d$',flags=0)

    dir_names_filtered = [d for d in dir_names if filter1.match(d) or filter2.match(d)]

    #verify the nums of case
    if len(dir_names_filtered) != nums:
        bpcase_caption = 'CaseNums Error'

    # sort the dir names  case01 ===> 1
    dir_names_filtered.sort(key= lambda x:int(x[4:]))
    # create paths list
    casepath_list = list(map(lambda x:os.path.join(bppath,x), dir_names_filtered))
    caseindex = list(map(lambda x:int(x[4:]),dir_names_filtered))

    # init a sheet book for saving the caption information for each csv files
    sheetbook = pd.DataFrame()
    # walk the list of [case1, case2, case3 ...]
    # check the details of case folder via using check_index_file function
    for index, casename,casepath in zip(caseindex, dir_names_filtered,casepath_list):

        value=''
        try:
            value = mainsheetbook.loc[bpname,f'case {index}']
        except Exception as e:
            pass

        if value == 'OK; ' and not csv_overwrite:
            print(f'Skip {bpname}: {casename}')
        else:
            # walk the list of [case1_1.csv, case1_2.csv, case1_3.csv ...]
            sheetbook,caption_list= check_index_file(sheetbook,casename,casepath,csv_rows,73)
            caption_str = ''
            for s in caption_list:
                caption_str = caption_str.join(f'{s}; ')
            mainsheetbook.loc[bpname,f'case {index}'] = caption_str

    mainsheetbook.loc[bpname,'CaseNums'] = bpcase_caption
    sheetbook.to_csv(os.path.join(savepath,f'{bpname}.csv'))

    return mainsheetbook


def check_index_file(sheetbook,casename,casepath,csv_rows,nums=73):
    caption_list=[]
    index_caption = 'OK'
    #read and sort the name of csv file
    csv_names = os.listdir(casepath)

    filter1 = re.compile('^case\d{1,2}_\d{1,2}.csv$',flags=0)
    filter2 = re.compile('^case\d{1,2}_\d{1,2}.csv$',flags=0)

    csv_names_filtered = [d for d in csv_names if filter1.match(d) or filter2.match(d)]

    csv_names = csv_names_filtered

    csv_names.sort(key=lambda x:int(x.split('_')[1][:-4]))
    csv_index = list(map(lambda x:int(x.split('_')[1][:-4]),csv_names))
    csv_name_paths = list(map(lambda x:os.path.join(casepath,x), csv_names))

    #check the nums of index
    if len(csv_index) != int(nums):
        index_caption = 'IndexNums Error'
    caption_list.append(index_caption)

    #check csv via check_csv function
    for csvpath, index in zip(csv_name_paths,csv_index):
        # check the details of csv file
        read_caption, nums_caption, velocity_caption = check_csv(csvpath, None, csv_rows, index)
        
        # write the captions of current csv file to the sheetbook
        caption_str=''
        for s in [read_caption, nums_caption, velocity_caption]:
            if s != 'OK':
                caption_str = caption_str.join(f'{s}; ')
        if len(caption_str) == 0:
            caption_str = 'OK'
        sheetbook.loc[casename,index] = f'{caption_str}'

        for c in [read_caption, nums_caption, velocity_caption]:
            caption_list.append(c)

    caption_list = list(set(caption_list))
    # remove 'OK' token
    if 'OK' in caption_list and len(caption_list) != 1:
        caption_list.remove('OK')

    return sheetbook, caption_list
        
def check_csv(csvpath1,csvpath2,rows,index):

    read_caption = 'OK'
    nums_caption = 'OK'
    velocity_caption = 'OK'

    data = None

    # read csv
    try:
        data = pd.read_csv(csvpath1, header= 2)
    except FileNotFoundError:
        try:
            data = pd.read_csv(csvpath2, header= 2)
        except Exception as e:
            print(f'    Error when read file  {csvpath2} :   {e}')
            read_caption = 'Read Error'
    except Exception as e:
        print(f'    Error when read file  {csvpath1} :   {e}')
        read_caption = 'Read Error'


    if read_caption == 'OK':

        print(f'Verifying the details of {csvpath1}...')

        if rows is not None:
            #verify the nums of row
            if len(data['I']) != int(rows):
                nums_caption = 'Rows Error'
        else:
            nums_caption = 'No BP Info'

        # verify the Velocity
        # when the nums of row is equal to variable [rows] and index is equal to 1
        if nums_caption == 'OK' and int(index) == 1:

            if max(data['Velocity (magnitude Max)']) > 0.0:
                velocity_caption = 'Velocity Error'

    return read_caption, nums_caption, velocity_caption

def run(rivername,inputfolder,outputfolder,tempfolder,
        meshinfo_file=r'破堤点格子番号.xlsx',
        zip_verify = True,
        csv_overwrite = False,
        extract_overwrite = False, 
        extract_remove= False):
    # check the paths
    for dir in [outputfolder,tempfolder]:
        if not os.path.exists(dir):
            os.makedirs(dir)


    name7z_List = os.listdir(inputfolder)
    # filter the unwanted files
    filter = re.compile('^BP\d\d\d.7z$',flags=0)
    name7z_List_filted = [f for f in name7z_List if filter.match(f) ]

    name7z_List = name7z_List_filted
    path7z_List = list(map(lambda x:os.path.join(inputfolder,x), name7z_List))
    nameBP_List = list(map(lambda x:x.split('.')[0], name7z_List))

    
    # load the xlsx file of mesh and drop the non value
    mesh = pd.read_excel(meshinfo_file,skiprows=0,index_col=0).dropna()

    for nameBP,name7z,path7z in zip(nameBP_List,name7z_List,path7z_List):

        # init the main sheetbook
        try:
            mainsheetbook = pd.read_csv(outputfolder+f'/{rivername}.csv',index_col=0)
        except Exception as e:
            print(f'When read the file of {rivername}.csv, error occurred: {e}')
            mainsheetbook = pd.DataFrame()

        # extract the .7z file
        extract_caption = extract_7z(path7z,name7z,tempfolder,zip_verify,extract_overwrite)

        mainsheetbook.loc[nameBP,'Extract Error'] = extract_caption
        #the dir of extracted BP
        BPpath = os.path.join(tempfolder,nameBP)

        # verify the extent of extracted files if extract operation succeeded
        if extract_caption == 'OK':
            try:
                # extract the info of I and J
                IJ = mesh.loc[nameBP].to_numpy()[1:3]
                csv_rows = int(IJ[0])*int(IJ[1])
            except Exception as e:
                print (f'No {nameBP} information in 破堤点格子番号.xlsx')
                csv_rows = None

            # check the details of BP
            mainsheetbook = check_bpcase(mainsheetbook,nameBP,BPpath,csv_rows,outputfolder,31,csv_overwrite)
            # save the mainsheetbook for current bp
            mainsheetbook.to_csv(outputfolder+f'/{rivername}.csv')

            if extract_remove :
                shutil.rmtree(os.path.join(tempfolder,nameBP))
                print(f'Remove {os.path.join(tempfolder,nameBP)}')

if __name__ == "__main__":
    river_list = ['Kinugawa','Kokaigawa']

    for river in river_list:
        # the path of XINJUN-PC
        input_path = f'F:\\Projects\\Flood\Kinugawa\\01_Raw_Data\\{river}' 
        output_path =  'F:\\Projects\\Flood\Kinugawa\\02_Deep_Learning\\Files Check Results'      
        temp_path = 'F:\\Projects\\Flood\Kinugawa\\02_Deep_Learning\\CasesData'

        # # the path on dl-box
        # input_path = f'F:\\氾濫予測AI\\01_収集資料\\{river}\\CasesData' 
        # output_path =  '../Files Check Results'      
        # temp_path = '../CasesData'

        run(river,input_path,output_path,temp_path,r'破堤点格子番号.xlsx',True,False,False,False)
    print('Done!')



    



