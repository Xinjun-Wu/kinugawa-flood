import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import datetime
from datetime import timedelta
import matplotlib.cm as cm
import glob
from PIL import Image
import time
import datetime
from tqdm import tqdm
from matplotlib import rcParams
import argparse
import sys



def customize_plot(target_value, predicted_value, title, figsize, dpi=100, max_value=4):

    config = {
    "font.family":'serif',
    "font.size": 8,
    "mathtext.fontset":'stix',
    "font.serif": ['Times New Roman'],
    }
    rcParams.update(config)

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=figsize, dpi=dpi)
    fig.suptitle(title)

    ax0 = axs[0]
    im0 = ax0.imshow(target_value,cmap =cm.jet,clim=(0,max_value))
    #ax0.axis('tight')
    #position=fig.add_axes([0.3, 0.05, 0.1, 0.03])
    cb0 = fig.colorbar(mappable=im0,ax=ax0)
    ax0.set_title('Target Water Depth')
    ax0.axis('off')

    ax1 = axs[1]
    im1 = ax1.imshow(predicted_value,cmap =cm.jet,clim=(0,max_value))
    #ax1.axis('tight')
    cb1 = fig.colorbar(mappable=im1,ax=ax1)
    ax1.set_title('Predicted Water Depth')
    ax1.axis('off')

    ax2 = axs[2]
    im2 = ax2.imshow(target_value-predicted_value,cmap=plt.get_cmap('RdBu'),clim=(-0.6,0.6))
    #ax2.axis('tight')
    cb2 = fig.colorbar(mappable=im2,ax=ax2)
    ax2.set_title('Difference Water Depth\n(Taget - Predicted)')
    ax2.axis('off')

    #plt.tight_layout()

    return fig

def data2csv(output_foldetr,data,step):
    depth_path = os.path.join(output_foldetr,'water depth')
    X_path = os.path.join(output_foldetr,'X flux')
    Y_path = os.path.join(output_foldetr,'Y flux')
    #Image_path = os.path.join(output_foldetr,'image')
    path_List = [depth_path,X_path,Y_path]
    
    for path in path_List:
        if not os.path.exists(path):
            os.makedirs(path)

    n_sample = data.shape[0]
    
    for n in range(n_sample):
        time_index = n + step
        sample_data = data[n]

        for c in range(3):
            channel_data = sample_data[c]
            savename = os.path.join(path_List[c],f'{time_index}.csv')
            np.savetxt(savename,channel_data,fmt="%.4f",delimiter=',')


def image2gif(input_folder,outputname):
    pngList = glob.glob(input_folder + "\*.png")
    images = []
    for png in pngList:
        im=Image.open(png)
        images.append(im)
    images[0].save(input_folder+os.sep+outputname, save_all=True, append_images=images, loop=0, duration=1000)

def result_output(inputpath,output_folder,step,casename,figsize,dpi,max_value):
    raw_data = np.load(inputpath)
    target_data = raw_data['input']
    predicted_data = raw_data['output']
    if not os.path.exists(os.path.join(output_folder,'image')):
        os.makedirs(os.path.join(output_folder,'image'))

    data2csv(output_folder,predicted_data,step)

    n_sample = target_data.shape[0]
    for n in range(n_sample):

        time_index = n + step
        time_stamp = str(timedelta(seconds=time_index*600) - timedelta(seconds=0))
        figtitle = f'Water Depth in {casename}\n {time_stamp}'

        fig = customize_plot(target_value=target_data[n][0], predicted_value=predicted_data[n][0],
                    title=figtitle,figsize=figsize,dpi=dpi,max_value=max_value)
        fig.savefig(os.path.join(output_folder,'image',f"{time_index}.png"))
        plt.close()

    
    #image2gif(output_folder, f'{casename}.gif')


if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('BPNAME')
    parser.add_argument('STEP')
    parser.add_argument("VERSION")
    parser.add_argument('EPOCH')
    parser.add_argument('CASE')
    args = parser.parse_args()
    parser.parse_args()

    BPNAME = args.BPNAME
    STEP = int(args.STEP)
    VERSION = int(args.VERSION)
    EPOCH = int(args.EPOCH)
    CASE = args.CASE

    # STEP = 6
    # VERSION = 1
    # EPOCH = 10
    # CASE = 'case2'

    FIGSIZE = (8,3)
    DPI = 100
    MAX_VALUE = 5

    INPUT_FOLDER = f"../Save/{BPNAME}/Step_{STEP}/test/model_V{VERSION}_epoch_{EPOCH}/{CASE}.npz"
    OUTPUT_FOLDER = f"../Save/{BPNAME}/Step_{STEP}/test/model_V{VERSION}_epoch_{EPOCH}/{CASE}/"
    print('Processing...')
    result_output(INPUT_FOLDER,OUTPUT_FOLDER,STEP,CASE,FIGSIZE,DPI,MAX_VALUE)
    print('Done!')



    













