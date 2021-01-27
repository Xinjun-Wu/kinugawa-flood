import os
from tqdm import tqdm
import py7zr
import pandas as pd
import re
import shutil

def extract_7z(path_7z,name_7z,extract_path,verify = True, overwrite = False):
    CRCS_value = 0
    caption = '-'

    if verify:
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
        # check the exists of extract path of current BP
        if not os.path.exists(os.path.join(extract_path,name_7z.split('.')[0])) or overwrite:

                print(f'Extracting {name_7z} ...')
                with py7zr.SevenZipFile(path_7z, 'r') as archive:
                    try:
                        archive.extractall(extract_path)
                        print(f'Extracte {name_7z} succeddfully!')
                    except Exception as e:
                        print(f'    Error when extract the {name_7z}:   {e}')
                        caption = 'Extract Error'

    return caption

def check_bpcase(bpname,bppath,csv_rows,savepath,nums=31):

    caption_list = []
    bpcase_caption = '-'
    # read and filter the name of case ,such as 'case1' or 'case01'
    dir_names = os.listdir(bppath)

    filter1 = re.compile('^case\d$',flags=0)
    filter2 = re.compile('^case\d\d$',flags=0)

    dir_names_filtered = [d for d in dir_names if filter1.match(d) or filter2.match(d)]

    #verify the nums of case
    if len(dir_names_filtered) != nums:
        bpcase_caption = 'CaseNums Error'

    caption_list.append(bpcase_caption)

    # sort the dir names  case01 ===> 1
    dir_names_filtered.sort(key= lambda x:int(x[4:]))
    # create paths list
    casepath_list = list(map(lambda x:os.path.join(bppath,x), dir_names_filtered))

    # init a sheet book for saving the caption information for each csv files
    sheetbook = pd.DataFrame()
    # check the details of case folder via using check_index_file function
    for casename,casepath in zip(dir_names_filtered,casepath_list):
        sheetbook= check_index_file(caption_list,sheetbook,casename,casepath,csv_rows,73)
    
    sheetbook.to_csv(os.path.join(savepath,f'{bpname}.csv'))

    return caption_list

def check_index_file(caption_list,sheetbook,casename,casepath,csv_rows,nums=73):
    index_caption = '-'
    #read and sort the name of csv file
    csv_names = os.listdir(casepath)
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
        sheetbook.loc[casename,index] = f'{read_caption}; {nums_caption}; {velocity_caption}'

        for c in [read_caption, nums_caption, velocity_caption]:
            caption_list.append(c)
    
    return sheetbook
        
def check_csv(csvpath1,csvpath2,rows,index):

    read_caption = '-'
    nums_caption = '-'
    velocity_caption = '-'

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


    if read_caption == '-':

        print(f'Verifying the details of {csvpath1}...')
        #verify the nums of row
        if len(data['I']) != int(rows):
            nums_caption = 'Rows Error'

        # verify the Velocity
        # when the nums of row is equal to variable [rows] and index is equal to 1
        if nums_caption == '-' and int(index) == 1:

            if max(data['Velocity (magnitude Max)']) > 0.0:
                velocity_caption = 'Velocity Error'

    return read_caption, nums_caption, velocity_caption

def run(rivername,inputfolder,outputfolder,tempfolder,meshinfo_file=r'破堤点格子番号.xlsx',
            verify = True, overwrite = False, remove=False):
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

    # init the main sheetbook
    mainsheetbook = pd.DataFrame()

    for nameBP,name7z,path7z in zip(nameBP_List,name7z_List,path7z_List):

        caption_list = []
        # extract the .7z file
        extract_caption = extract_7z(path7z,name7z,tempfolder,verify,overwrite)
        caption_list.append(extract_caption)
        #the dir of extracted BP
        BPpath = os.path.join(tempfolder,nameBP)

        # verify the extent of extracted files if extract operation succeeded
        if extract_caption == '-':
            # extract the info of I and J
            IJ = mesh.loc[nameBP].to_numpy()[1:3]
            csv_rows = int(IJ[0])*int(IJ[1])
            # check the details of BP
            caption_list = check_bpcase(nameBP,BPpath,csv_rows,outputfolder,31)
            caption_list.append(extract_caption)
            # remove the repeted items
            caption_list = list(set(caption_list))

            if remove :
                shutil.rmtree(os.path.join(tempfolder,nameBP))
                print(f'Remove {os.path.join(tempfolder,nameBP)}')

        # remove - token
        if '-' in caption_list and len(caption_list) != 1:
            caption_list.remove('-')

        ErrorInformation = ''
        for item in caption_list:
            ErrorInformation += f'{item}; '

        mainsheetbook.loc[nameBP,'Error Information'] = ErrorInformation
        mainsheetbook.to_csv(outputfolder+f'/{rivername}.csv')



if __name__ == "__main__":
    river_list = ['Kinugawa','Kokaigawa']

    for river in river_list:
        input_path = f'F:\\Projects\\Flood\Kinugawa\\01_Raw_Data\\{river}' 
        output_path =  'F:\\Projects\\Flood\Kinugawa\\02_Deep_Learning\\Files Check Results'      
        temp_path = 'F:\\Projects\\Flood\Kinugawa\\02_Deep_Learning\\CasesData'

        run(river,input_path,output_path,temp_path,)
    print('Done!')



    



