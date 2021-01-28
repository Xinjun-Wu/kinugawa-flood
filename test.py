# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import numpy.ma as ma
import torch


# %%
data_a = np.array([[0,0,0],[10.2,0.1,0.1],[0,0,0]])
data_b = np.array([[0.1,0.1,0.1],[0.1,0.1,0.1],[0.1,0.1,0.1]])


# %%
masked_a = ma.masked_equal(data_a,0)
masked_a


# %%
masked_b = ma.masked_array(data_b,masked_a.mask,fill_value = 0)
masked_b


# %%
masked_b.filled()


# %%
z = masked_a.mask


# %%
(n_i, n_j) = z.shape


# %%
for i in range(n_i):
    for j in range(n_j):
        if z[i,j] == False:
            print(z[i,j])


# %%
to = torch.tensor(masked_b)


# %%
to


# %%
def search_edge(input,output,point_i,point_j, height, width):
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
    
def mask_buffer(input_mask,height,width):
    #根据原始mask的边缘，通过缓冲区创建更大大的mask
    output_mask = input_mask.copy()
    for i in range(input_mask.shape[0]):
        for j in range(input_mask.shape[1]):
            search_edge(input_mask,output_mask,i,j,height,width)
    return output_mask

def area_extract(target_area, extract_area, buffer_height,buffer_width):
    #通过目标区域的buffered mask 提取下一时刻的研究区域
    #目标区域的掩码识别值为0，当值为0时，掩码值为True.否则为False
    #buffer 方式为矩形buffer,根据每一个边缘点的矩形范围创建buffer
    target_masked = ma.masked_equal(target_area,0)
    input_mask = target_masked.mask
    buffered_mask = mask_buffer(input_mask,buffer_height,buffer_width)
    extracted_area = ma.masked_array(extract_area, buffered_mask, fill_value = 0)
    extracted_area = extracted_area.filled()
    return extracted_area
    


# %%
a = torch.zeros((10,10))
a[4:6,0:3] = 5
a


# %%
b = torch.ones((10,10))
b


# %%
b_ = area_extract(a,b,1,1)
b_


# %%



# %%
import numpy as np


# %%
a = np.array([1,2,3,4])


# %%
a = a.reshape(2,2)


# %%
a


# %%
b = np.tile(a,(3,1,1,1))


# %%
b


# %%
b.shape


# %%



