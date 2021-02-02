############################### Kinugawa-flood ##############################

channel
    *master                     <发布稳定版本，能够在九大或者东京机器上稳定运行>
    *alpha-cooperate            <根据合作研究需求进行的开发分支>
    *alpha-dev                  <可能的下一步开发分支>
    *beta-academic              <根据论文要求与想法进行的开发分支>
    *feature                    <添加新功能时先在feature分支上实现，
                                 并测试通过后再合并到其他分支>
    *bugfix                     <此分支为bug分支，当存在bug时，
                                 在此分支解决后再布署到其他相对
                                 的分支>

##############################################################################

############################## Running Steps #################################

Step 1 <colne the repo from GitHub>
        When you browse the current page, you could colne the repo from this page
        
Step 2 <create the enviroment in conda or miniconda>
