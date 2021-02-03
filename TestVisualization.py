
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
from mpl_toolkits.axes_grid1 import AxesGrid
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

    para_data =data = np.array([target_value, predicted_value, target_value-predicted_value])
    para_figsize = figsize
    para_dpi = dpi
    n_row = 3
    n_col = 3

    row_ylabel = ['WaterDepth(m)', 'Velocity(ms-1)X', 'Velocity(ms-1)Y']
    #row_xlabel = ['a','b','c']
    col_titles = ['Target', 'Prediction', 'Target-Prediction']


    fig = plt.figure(figsize=para_figsize, dpi=para_dpi)
    fig.suptitle(title)

    grid = AxesGrid(fig, (0.05,0.05,0.9,0.89),  # similar to subplot(122)
                    nrows_ncols=(n_row, n_col),
                    axes_pad=0.3,
                    label_mode="all",
                    share_all=True,
                    cbar_location="right",
                    cbar_mode="each",
                    cbar_size="7%",
                    cbar_pad="10%",
                    )

    for col in range(n_col): #target, prediction, target-prediction
        for row in range(n_row): # 遍历三通道
            plot_data = para_data[col,row] 

            i = row*n_col+col
            ax = grid[i]

            if col != 2 and row == 0:
                im = ax.imshow(plot_data, cmap=cm.jet, clim=(0,5),aspect = 0.4) # 水深的目标值与预测值的分布图
            elif col == 2 and row == 0:
                im = ax.imshow(plot_data, cmap=plt.get_cmap('RdBu'), clim=(-0.5,0.5),aspect = 0.4) # 水深的误差值的分布图
            elif col!= 2 and row != 0:
                im = ax.imshow(plot_data, cmap=cm.jet, clim=(-1,1),aspect = 0.35) # 流速的目标值与预测值的分布图
            else:
                im = ax.imshow(plot_data, cmap=plt.get_cmap('RdBu'), clim=(-0.1,0.1),aspect = 0.4) # 流速的误差的分布图
            #ax.set_axis_off()
            ax.tick_params(bottom=False,top=False,left=False,right=False,
            labelbottom=False, labeltop=False, labelleft=False, labelright=False)
            #ax.set_xlabel(row_xlabel[row]+f'({col+1})')
            #fig.savefig('../comparison.png')
            # if i % n_col:
            #     cb = grid.cbar_axes[i//n_col].colorbar(im)
            #     cb.ax.tick_params(direction='in',size=1)
            cb = grid.cbar_axes[i].colorbar(im)
            cb.ax.tick_params(direction='in',size=1)

            if row == 0:
                ax.set_title(f'{col_titles[col]}')

            if col == 0:
                ax.set_ylabel(row_ylabel[row])

    # fig, axs = plt.subplots(nrows=1, ncols=3, figsize=figsize, dpi=dpi)
    # fig.suptitle(title)

    # ax0 = axs[0]
    # im0 = ax0.imshow(target_value,cmap =cm.jet,clim=(0,max_value))
    # #ax0.axis('tight')
    # #position=fig.add_axes([0.3, 0.05, 0.1, 0.03])
    # cb0 = fig.colorbar(mappable=im0,ax=ax0)
    # ax0.set_title('Target Water Depth')
    # ax0.axis('off')

    # ax1 = axs[1]
    # im1 = ax1.imshow(predicted_value,cmap =cm.jet,clim=(0,max_value))
    # #ax1.axis('tight')
    # cb1 = fig.colorbar(mappable=im1,ax=ax1)
    # ax1.set_title('Predicted Water Depth')
    # ax1.axis('off')

    # ax2 = axs[2]
    # im2 = ax2.imshow(target_value-predicted_value,cmap=plt.get_cmap('RdBu'),clim=(-0.6,0.6))
    # #ax2.axis('tight')
    # cb2 = fig.colorbar(mappable=im2,ax=ax2)
    # ax2.set_title('Difference Water Depth\n(Taget - Predicted)')
    # ax2.axis('off')

    # #plt.tight_layout()

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
    #pngList.sort(key=lambda x:int(x.split('/')[-1][:-4]), reverse=False)
    images = []
    for png in pngList:
        im=Image.open(png)
        images.append(im)
    images[0].save(input_folder+os.sep+outputname, save_all=True, append_images=images, loop=0, duration=500)

def result_output(inputpath,output_folder,step,casename,figsize,dpi,max_value):
    raw_data = np.load(inputpath)
    target_data = raw_data['input']
    target_data = target_data.squeeze()
    predicted_data = raw_data['output']
    predicted_data = predicted_data.squeeze()
    if not os.path.exists(os.path.join(output_folder,'image')):
        os.makedirs(os.path.join(output_folder,'image'))

    data2csv(output_folder,predicted_data,step)

    n_sample = target_data.shape[0]
    for n in range(n_sample):

        time_index = n + step
        time_stamp = str(timedelta(seconds=time_index*600) - timedelta(seconds=0))
        figtitle = f' case {casename} in {time_stamp}'

        fig = customize_plot(target_value=target_data[n], predicted_value=predicted_data[n],
                    title=figtitle,figsize=figsize,dpi=dpi,max_value=max_value)
        fig.savefig(os.path.join(output_folder,'image',f"{time_index:03}.png"))
        plt.close()

    
    image2gif(output_folder+'/image', f'{casename}.gif')


if __name__ == '__main__':

    BRANCH = 'Master Branch'
    # BRANCH = 'alpha-academic Branch'
    # BRANCH = 'alpha-cooperate Branch'
    # BRANCH = 'beta-dev Branch'

    # parser = argparse.ArgumentParser()
    # parser.add_argument('BPNAME')
    # parser.add_argument('STEP')
    # parser.add_argument("VERSION")
    # parser.add_argument('EPOCH')
    # parser.add_argument('CASE')
    # args = parser.parse_args()
    # parser.parse_args()

    # BPNAME = args.BPNAME
    # STEP = int(args.STEP)
    # VERSION = int(args.VERSION)
    # EPOCH = int(args.EPOCH)
    # CASE = args.CASE

    ACADEMIC = False

    GROUP_ID = 'Ki1'
    #ID_item = GROUP_ID
    ID_item_list = [
                'BP020',
                'BP032',
                'BP022',
                'BP025',
                'BP028',
                'BP031',
                'BP037',
                'BP040',
                    ]   

    CASEINDEX_list = ['_006','_014','_023','_031']


    STEP = 'Step_01'
    VERSION = 1
    EPOCH = 11000


    for ID_item in ID_item_list:
        for CASEINDEX in CASEINDEX_list:

            CASENAME = ID_item+CASEINDEX

            FIGSIZE = (5,10)
            DPI = 100
            MAX_VALUE = 5

            
            INPUT_FOLDER = f'../Save/{BRANCH}/TestResults/{STEP}/{ID_item}/model_V{VERSION}_epoch_{EPOCH}/{CASENAME}.npz'
            OUTPUT_FOLDER = f'../Save/{BRANCH}/TestResults/{STEP}/{ID_item}/model_V{VERSION}_epoch_{EPOCH}/{CASENAME}/'
            print(f'Processing for {CASENAME}...')
            result_output(INPUT_FOLDER,OUTPUT_FOLDER,1,CASENAME,FIGSIZE,DPI,MAX_VALUE)
    
    print('Done!')



    













