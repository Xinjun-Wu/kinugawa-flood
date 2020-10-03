# kinugawa-flood
新的版本

submit 001 <add/delete/optimize>
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


