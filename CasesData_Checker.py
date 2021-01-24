import os
import pandas as pd
import numpy as np


def BPcase_checker(summary_pd,BPNAME,BPpath,casestart = 1, caseend = 31, nums = 31):
    for i in np.linspace(start = casestart,stop=caseend, num=nums):
        # the names of each case
        casename = f'case{int(i)}' # case1
        casename_other = f'case{int(i):02}' # case01

        if os.path.exists(os.path.join(BPpath,casename)):
            pass # check the details of the current case
        elif os.path.exists(os.path.join(BPpath,casename_other)):
            pass # check the details of the current case
        else:
            print(f'There is no such folder: {os.path.join(BPpath,casename)}')
            print(f'There is no such folder: {os.path.join(BPpath,casename_other)}\r\n')

            summary_pd.loc[BPNAME][casename] = 'No folder'

    return summary_pd

def case_details_checker(casename,casepath,indexstart = 1, indexend = 73,nums = 73):
    for i in np.linspace()


if __name__ == '__main__':
    CHECKER_RESULT = '../CasesData/checker results.xlsx'

    if not os.path.exists


