import numpy as np
import numpy.ma as ma
from sklearn import preprocessing
import torch
from torch import tensor
from torch._C import device

def _search_edge(input,output,point_i,point_j, height, width):
    #识别边缘的点，并将缓冲区域内的点值更改为False
    if point_i - height < 0:
        start_i = 0
    else:
        start_i = point_i - height
    if point_j - width < 0:
        start_j = 0
    else:
        start_j = point_j - width
    if point_i + height+1 >= input.shape[0]:
        end_i = -1
    else:
        end_i =  point_i + height+1
    if point_j + width+1 >= input.shape[1]:
        end_j = -1
    else:
        end_j =point_j + width+1
    output[point_i,point_j] = input[start_i:end_i,start_j:end_j].all()
    
def _mask_buffer(input_mask,height,width):
    #根据原始mask的边缘，通过缓冲区创建更大大的mask
    output_mask = input_mask.copy()
    for i in range(input_mask.shape[0]):
        for j in range(input_mask.shape[1]):
            _search_edge(input_mask,output_mask,i,j,height,width)
    return output_mask

def area_extract(target_area, extract_area, buffer_height, buffer_width, equal_value, less_equal_value):
    """ #输入为二阶array或者tensor 
        #通过目标区域的buffered mask 提取下一时刻的研究区域
        #目标区域的掩码识别值为equal_value，当值为等于mask_value时，掩码值为True.否则为False
        #目标区域的掩码识别值为less_equal_value，当值为小于等于mask_value时，掩码值为True.否则为False
        #less_equal_value和equal_value二选一,另一个值输入为None
        #buffer 方式为矩形buffer,根据每一个边缘点的矩形范围创建buffer
        # 
    """
    if isinstance ( target_area, torch.Tensor):
        target_area = target_area.cpu()
        target_area = target_area.numpy()

    if isinstance (extract_area, torch.Tensor):
        extract_area = extract_area.cpu()
        extract_area = extract_area.numpy()

    if equal_value is not None and less_equal_value is None:
        target_masked = ma.masked_equal(target_area, equal_value)
    elif equal_value is None and less_equal_value is not None:
        target_masked = ma.masked_less_equal(target_area, less_equal_value)
    else:
        raise ValueError('equal_value 和 less_equal_value 其一必须为None')


    input_mask = target_masked.mask
    if buffer_height==0 and buffer_width ==0:
        buffered_mask = input_mask
    else:
        buffered_mask = _mask_buffer(input_mask,buffer_height,buffer_width)
        
    extracted_area = ma.masked_array(extract_area, buffered_mask, fill_value = 0)
    extracted_area = extracted_area.filled()
    return extracted_area


class data_decorater():
    def __init__(self,dem_path):
        raw_data = np.load(dem_path,allow_pickle=True)
        DEM = raw_data['DEM'][0]

        #对DEM进行归一化处理，采用MinMaxScaler
        scaler = preprocessing.MinMaxScaler()
        DEM_MinMax = scaler.fit_transform(DEM)

        #转化为array并拓展为四阶Array,并于后面broadcast
        DEM_MinMax_Array = np.array(DEM_MinMax)
        self.DEM = DEM_MinMax_Array
        self.n_add_channel = 1

    def add_dem(self,input_tensor):
        n_sample = input_tensor.size()[0]
        DEM_MinMax_Array = np.tile(self.DEM,(n_sample,1,1,1))
        target_device = input_tensor.device
        DEM_MinMax_Tensor = torch.tensor( DEM_MinMax_Array,device = target_device)
        output_tensor = torch.cat((DEM_MinMax_Tensor,input_tensor),1)#在通道上连接，DEM数据在前面
        return output_tensor
