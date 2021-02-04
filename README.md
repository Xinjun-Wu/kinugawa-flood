############################### Kinugawa-flood ##############################

branches
        # master >>>>>>>>>>>>>>>>>>     发布稳定版本，能够在九大或者东京机器上稳定运行;
        # alpha-cooperate >>>>>>>>>     根据合作研究需求进行的开发分支;
        # alpha-dev >>>>>>>>>>>>>>>     可能的下一步开发分支;
        # beta-academic >>>>>>>>>>>     根据论文要求与想法进行的开发分支;
        # feature >>>>>>>>>>>>>>>>>     添加新功能时先在feature分支上实现，
                                        并测试通过后再合并到其他分支;
        # bugfix >>>>>>>>>>>>>>>>>>     此分支为bug分支，当存在bug时，
                                        在此分支解决后再布署到其他相对分支

##############################################################################

############################## Running Steps #################################

Notice: Before Running those scripts, please make sure that you have prepared the
        essential deep leearning env featured with pytorch and the others packages, 
        like scipy, scikit-learn, tqdm and so on. 

        This document assume that you are familiar with the basic concepts of GitHub 
        technologies, such as concepts of fork, pull, commit, push, pull request and 
        issue. click https://docs.github.com/en if you need.

        If you enccounter some bugs or have some suggestions, please create issue or 
        pull request.

Step 1  Csv2Npy.py

        % params %
        {
                BPNAME_List = ['BP021','BP028]          # the list of name str of each BP
                INPUT = f'../CasesData/{BPNAME}'        # the input folder of each BP, optional 
                                                        # paramter. We recommend don't change it
                                                        # but put the CasesData folder in the same
                                                        # path with the current repo. 
                                                        # the directory tree can be seen as below.
                                                        # ..
                                                        # .|CasesData/
                                                        #  |     |BP028/
                                                        #  |     |BP027/
                                                        #  |     |...
                                                        # .|kinugawa-flood/
                                                        #  |     |Csv2Npy.py
                                                        #  |     |...
                                                        # .|Save
                                                        #  |     |Master Branch/
                                                        #  |     |...
        }

        # Please make sure that all case data is ready and any error is not included,
        # In general, each BP have 31 cases and 72 csv files are coverd by each case.
        # Besides, we suggest that the initial velocity should not equal to zero. If 
        # any BP have that velocity error, please ignore that the whole BP until the
        # all cases data are corrected.

        # Meanwhile, please fill the row sheet of 破堤点格子番号.xlsx file in current path, if the information 
        # you need was blank
        
        
Step 2 generateData.py

        % params %
        {
                BPNAME_List = ['BP021','BP028]          # the list of name str of each BP
                GROUP_ID = 'Ki1'                        # the str name of the group that
                                                        # above BP belong to.
                                                        # the information of Group can be 
                                                        # found in the 破堤点格子番号.xlsx 
                                                        # file.
        }

Step 3 TrainMethod_SingleBP.py

        % params %
        {
                GROUP_ID_List=[                         # the list of group id
                        'Ki1',
                        'Ki2',
                        ]
                BPNAME_ListofList = [                   # the cooredponding BP that belong 
                        ['BP021','BP028'],              # to GROUP_ID_List, 'BP021' and 'BP028'
                        ['BP008'],                      # belong to Ki1, 'BP008' belong to Ki2
                                ]
                CHECKPOINT_Dic = {                      # the checkpoint of each BP, None or list
                        'BP028':['Ki1', 1, 100]         # ['Ki1', 1, 100] means [GROUP, STEP, EPOCH]
                        'BP008':None,                   # None means no checkpoint, train it from 0
                                }
                EXCEPT_CASE_Dic = {                     # ignore the cases when tran the BPs,
                                'BP021':['BP021_006',   # None or list
                                        'BP021_014',    #
                                        'BP021_023',
                                        'BP021_031'], 
                                'BP033':None, 
                                }
        TRAIN_PARAMS_DICT = {                           # the list of parameters that used in train process
                'EPOCHS' : 6000,                        # the epoch that specificed in train loops
                'BATCHSIZES' : 128,                     # change it according to the memory of your GPU
                'MODEL_SAVECYCLE' : [
                                [2000,500], # the trained model will be Saved periodicaly on Epoch
                                [4000,200], # [2000,500] means the model will be saved in each 200 epoch
                                [5800,100], # between epoch 2000 and 4000, like 2200 and 2400, please
                                [6000,10],  # notify that the 4000 can be divided by 200 without remainder
                                #[2000,10], # and the last epoch must equal to the value of you set 
                                            # in 'EPOCHS' ,like 6000
                                ],
                'RECORDER_SAVECYCLE' :[
                                [2000,500], # the principle is same as above, 
                                [4000,200], # we suggest you keep the value same with 'MODEL_SAVECYCLE',
                                [5800,100], # thouhgt the value can be changed in the principle
                                [6000,10],  # 
                                #[2000,10], # 
                                ],
                ... other parameters        # don't need to change
        }

Step 4 TestMethod.py

        % params %
        {
                GROUP_ID_List = ['Ki1']         # the name of group, allowed to set only one group 
                                                # due to the old edition that i write, if you add a second
                                                # group name, the script may be encounter error.
                BP_ID_List = [
                                'BP020',        # the BP name that you want to test
                                'BP032',        # all the elements in list are belong to group
                                'BP022',        # that seted in GROUP_ID_List, like Ki1
                                'BP025',
                                'BP028',
                                'BP031',
                                'BP037',
                                'BP040',
                                ] 
                START_EPOCH =10990              # the range of epoch, start point not included.
                END_EPOCH = 11000               # 10990 will not be tested, but 11000 will.
                EPOCH_STEP = 10                 # don't change it in avoid of unpredictable errors

                TEST_CASE_LIST = ['_006','_014','_023','_031']  # those cases will be tested
                                                                # all BP share the varible
                                                                # please keep same with EXCEPT_CASE_Dic
                                                                # in step 3
        }

Step 5 TestMethod.py    # Visualize the results of test

        % params %
        {
                GROUP_ID = 'Ki1'        # the name of group
                ID_item_list = [        # the BP name that you want to visualize
                        'BP020',        # all the elements in list are belong to group
                        'BP032',        # that seted in GROUP_ID, like Ki1
                        'BP022',
                        'BP025',
                        'BP028',
                        'BP031',
                        'BP037',
                        'BP040',
                        ]   
                CASEINDEX_list = ['_006','_014','_023','_031']  # those cases will be visualized
                                                                # all BP share the varible
                                                                # please keep same with EXCEPT_CASE_Dic
                                                                # in step 3
                EPOCH = 11000                                   # please specific the target epoch
                                                                # the results of target epoch will be
                                                                # draw as a sequences of plots
        }
############################## Running Steps End #################################
吴 鑫俊
bent20140204@outlook.com
