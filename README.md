############################### Kinugawa-flood ##############################

channel
    *master                     <发布稳定版本，能够在九大或者东京机器上稳定运行>
    *alpha-cooperate            <根据合作研究需求进行的开发分支>
    *beta-academic              <根据论文要求与想法进行的开发分支>
    *feature                    <添加新功能时先在feature分支上实现，
                                 并测试通过后再合并到其他分支>
    *bugfix                     <此分支为bug分支，当存在bug时，
                                 在此分支解决后再布署到其他相对
                                 的分支>

##############################################################################

submit 005 <optimize>
2020-10-17
    optimize
    Csv2Npy.py, TrainAndTest.py, TrainMethod.py 参数传递
    # kinugawa-flood

submit 004 <optimize>
2020-10-15
    optimize 
        修改文件目录的组织方式及命名方式，所有BP的case放在同一个文件夹，通过文件名进行定位；
        训练时可选择不参与训练的BP或者CASE;
        其他优化

optimize
    results_output.py 优化生成图片的排版
     其他关联tools.py相应的更改

submit 003 <add/optimize>
2020-10-06 

add 
    tootls.py 生成掩膜时，自动将张量输入转化为numpy array
                两种淹没生成方式，less or less_equal

submit 002 <debug>
2020-10-03 
debug 
    TrainAndTest.py 在test时或者将来的预测时，初始时刻使用破堤点作为目标区域生成 buffered mask, 其他时刻使用上一时刻的洪水范围

submit 001 <add/debug/delete/optimize>
2020-10-03 
add
    Csv2Npy.py 添加忽略文件及文件夹功能，在遍历所有case列表时，自动跳过 except_list 内的条目。

    generateData.py 通过水流速度的范围对水深区域的值进行提取，非水流泛滥区域的水深值填充为0；
                    对入流流量的值进行小时定标标准化，通过原有数值除以定值10000000(一千万)实现。

    TrainAndTest.py 数据参与训练或者测试时添加DEM等固定的数据；
                    对预测的数据进行缓冲后的淹没处理，剔除非研究区域的错误预测值的影响。

    TrainMethod.py 训练前传入数据装饰器类的实例，用于在训练与测试时添加DEM等数据。

delete
    generateData.py 移除添加DEM数据的通道

optimize 
    细节优化
