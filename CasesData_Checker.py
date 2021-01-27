import os
import pandas as pd
import numpy as np
from tqdm import tqdm


def BPcase_checker(excel_writer,summary_pd,pointlists,BPNAME,BPpath,casestart = 1, caseend = 31, nums = 31):
    """
    fouce on checking the nums of case folder
    the details of case folder and the case files can be implemented by the sub functions 

    """
    #array([284.0, 99.0, 195.0, 196.0, 197.0, 198.0, 199.0], dtype=object)
    if BPNAME in pointlists.index:
        try:
            pointsinfo_array = pointlists.loc[BPNAME].to_numpy()[1:8] #[ROW,COLUMNS,POINT1,POINT2,POINT2,POINT4]
        except Exception as e:
            print(f"when extract the points of {BPNAME} with file: 破堤点格子番号.xlsx")
            print(f'     {e}')
            pointsinfo_array = None

    sub_sheet_pd = pd.DataFrame()

    for i in tqdm(np.linspace(start = casestart,stop=caseend, num=nums)):
        # the names of each case
        casename = f'case{int(i)}' # case1
        casepath = os.path.join(BPpath,casename)

        if os.path.exists(casepath):# check the details of the current case
            sub_sheet_pd,caption_str = case_nums_checker(pointsinfo_array,sub_sheet_pd,casename,casepath,1,73,73)
            summary_pd.loc[BPNAME,casename] = caption_str
        else:
            print(f'There is no such folder: {casepath}')
            summary_pd.loc[BPNAME,casename] = 'No folder'

    sub_sheet_pd.to_excel(excel_writer,sheet_name=BPNAME)
    #excel_writer.save()  # 储存文件

    return summary_pd

def case_nums_checker(pointsinfo,sub_sheet_pd, casename,casepath,indexstart = 1, indexend = 73,nums = 73):
    """
    checking the nums of files in the target case folder
    for example the nums of BP028_case1 is equal to 73
    """
    caption_List = [] #caption information will be displayed in the main sheet book
    for i in np.linspace(start = indexstart, stop=indexend,num=nums):
        indexname = str(int(i))
        csv_file_path = os.path.join(casepath,f'{casename}_{indexname}.csv')
        if os.path.exists(csv_file_path):
            details_info = casefile_details_checker(pointsinfo,csv_file_path,indexname) #check the details of the casd index file
        else:
            details_info = 'No file'
        
        sub_sheet_pd.loc[casename, indexname] = details_info

        if details_info not in caption_List:
            caption_List.append(details_info)

    # checking the extent of caption list
    if len(caption_List) == 1 and '-' in caption_List :
        caption_str = '-'
    else:
        if '-' in caption_List:
            caption_List.remove('-')
        caption_str = ''
        for item in caption_List:
            caption_str += (f'{item}; ')
        
    return sub_sheet_pd, caption_str

def casefile_details_checker(pointsinfo,csvfile,index):

    if pointsinfo not None:
        nums = int(pointsinfo[0]*pointsinfo[1])
    else:
        nums = None
    try:
        files_pd = pd.read_csv(csvfile, header= 2)

        if int(index) == 1 and max(files_pd['Velocity (magnitude Max)']) > 0.0:
            return 'initial Velocity not 0'
        elif nums is None:
            return 'No points info in excel'
        elif nums != len(files_pd['I']):
            return 'incomplete file'
        else:
            return '-'
    except Exception as e:
        print(f'\r\n when read file from :{csvfile},\r\n    {e}')
        return 'read error'


if __name__ == '__main__':

    # CHECKER_RESULT = '../CasesData/checker results.xlsx'

    # BPNAME_List = os.listdir('../CasesData')
    # if 'checker results.xlsx' in BPNAME_List:
    #     BPNAME_List.remove('checker results.xlsx')

    # pointlists = pd.read_excel(r'破堤点格子番号.xlsx',skiprows=0,index_col=0).dropna()

    # with pd.ExcelWriter(CHECKER_RESULT) as excel_writer:
    #     main_sheetbook_pd = pd.DataFrame()

    #     for BPNAME in BPNAME_List:
    #         BP_PATH = os.path.join('../CasesData', BPNAME)
    #         print(f'Checking for {BPNAME}')
    #         main_sheetbook_pd = BPcase_checker(excel_writer,main_sheetbook_pd,pointlists,BPNAME,BP_PATH,1,31,31)
        
    #     main_sheetbook_pd.to_excel(excel_writer,sheet_name='main')

    #     print('Done')
    print('Please run check_all_data.py')









