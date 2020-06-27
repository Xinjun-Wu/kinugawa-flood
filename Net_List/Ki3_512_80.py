# coding: utf-8



import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from torch.autograd import Variable

###################

#NET FOR MESH 512*80

###################

# define the class of CNN
# a forward CNN
class ConvNet_2(nn.Module):
    def __init__(self,channel_n):
        super(ConvNet_2, self).__init__()
        
        # 5*512*80 - 16*508*76,
        self.convlayer1 = nn.Sequential(
            nn.Conv2d(channel_n,32, kernel_size = 5),
            nn.BatchNorm2d(32),
            nn.PReLU(32))
        if torch.cuda.device_count()>1:
            self.convlayer1 = nn.DataParallel(self.convlayer1)
        
        # NIN layer added
        self.convlayer1_1 = nn.Sequential(
            nn.Conv2d(32,32, kernel_size = 1),
            nn.BatchNorm2d(32),
            nn.PReLU(32))
        if torch.cuda.device_count()>1:
            self.convlayer1_1 = nn.DataParallel(self.convlayer1_1)
        
        # 32**508*76 - 64*254*38,
        self.convlayer2 = nn.Sequential(
            nn.Conv2d(32,64, kernel_size = 2, stride = 2),
            nn.BatchNorm2d(64),
            nn.PReLU(64))
        if torch.cuda.device_count()>1:
            self.convlayer2 = nn.DataParallel(self.convlayer2)
        # NIN layer added
        self.convlayer2_1 = nn.Sequential(
            nn.Conv2d(64,64, kernel_size = 1),
            nn.BatchNorm2d(64),
            nn.PReLU(64))
        if torch.cuda.device_count()>1:
            self.convlayer2_1 = nn.DataParallel(self.convlayer2_1)
        
        # 64*254*38 - 256*127*19
        self.convlayer3 = nn.Sequential(
            nn.Conv2d(64,256, kernel_size = 2, stride = 2),
            nn.BatchNorm2d(256),
            nn.PReLU(256))
        if torch.cuda.device_count()>1:
            self.convlayer3 = nn.DataParallel(self.convlayer3)
        
        # NIN layer added
        self.convlayer3_1 = nn.Sequential(
            nn.Conv2d(256,256, kernel_size = 1),
            nn.BatchNorm2d(256),
            nn.PReLU(256))
        if torch.cuda.device_count()>1:
            self.convlayer3_1 = nn.DataParallel(self.convlayer3_1)
            
        # 256*127*19 - 256*127*19
        self.convlayer4 = nn.Sequential(
            nn.Conv2d(256,256, kernel_size = 3, padding = 1),
            nn.Conv2d(256,256, kernel_size = 1),
            nn.BatchNorm2d(256),
            nn.PReLU(256))
        if torch.cuda.device_count()>1:
            self.convlayer4 = nn.DataParallel(self.convlayer4)
        
        # 256*127*19 - 256*127*19
        self.convlayer5 = nn.Sequential(
            nn.Conv2d(256,256, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(256),
            nn.PReLU(256))
        if torch.cuda.device_count()>1:
            self.convlayer5 = nn.DataParallel(self.convlayer5)
        
        # 256*127*19 - 256*127*19
        self.convlayer6 = nn.Sequential(
            nn.Conv2d(256,256, kernel_size = 3, padding = 1),
            nn.Conv2d(256,256, kernel_size = 1),
            nn.BatchNorm2d(256),
            nn.PReLU(256))
        if torch.cuda.device_count()>1:
            self.convlayer6 = nn.DataParallel(self.convlayer6)
        
        # 256*127*19- 256*127*19
        self.convlayer7 = nn.Sequential(
            nn.Conv2d(256,256, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(256),
            nn.PReLU(256))
        if torch.cuda.device_count()>1:
            self.convlayer7 = nn.DataParallel(self.convlayer7)
        
        # 256*127*19 - 128*254*38
        self.convlayer8 = nn.Sequential(
            nn.ConvTranspose2d(256,128, kernel_size = 2, stride = 2),
            nn.BatchNorm2d(128),
            nn.PReLU(128))
        if torch.cuda.device_count()>1:
            self.convlayer8 = nn.DataParallel(self.convlayer8)
        
        
        # 128*254*38 - 64*508*76
        self.convlayer9 = nn.Sequential(
            nn.ConvTranspose2d(128,64, kernel_size = 2, stride = 2),
            nn.BatchNorm2d(64),
            nn.PReLU(64))
        if torch.cuda.device_count()>1:
            self.convlayer9 = nn.DataParallel(self.convlayer9)
        
        # NIN layer
        self.convlayer9_1 = nn.Sequential(
            nn.ConvTranspose2d(64,64, kernel_size = 1),
            nn.BatchNorm2d(64),
            nn.PReLU(64))
        if torch.cuda.device_count()>1:
            self.convlayer9_1 = nn.DataParallel(self.convlayer9_1)
        
        
        # 64*508*76 - 32*512*80
        self.convlayer10 = nn.Sequential(
            nn.ConvTranspose2d(64,32, kernel_size = 5),
            nn.BatchNorm2d(32),
            nn.PReLU(32))
        if torch.cuda.device_count()>1:
            self.convlayer10 = nn.DataParallel(self.convlayer10)
        
        # NIN
        self.convlayer10_1 = nn.Sequential(
            nn.ConvTranspose2d(32,32, kernel_size = 1),
            nn.BatchNorm2d(32),
            nn.PReLU(32))
        if torch.cuda.device_count()>1:
            self.convlayer10_1 = nn.DataParallel(self.convlayer10_1)
        
        # 32*512*80 - 12*512*80
        self.convlayer11 = nn.Sequential(
            nn.Conv2d(32,12, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(12),
            nn.PReLU(12))
        if torch.cuda.device_count()>1:
            self.convlayer11 = nn.DataParallel(self.convlayer11)
        
        # 12*512*80 - 3*512*80
        self.convlayer12 = nn.Sequential(
            nn.Conv2d(12,3, kernel_size = 1),
            nn.BatchNorm2d(3),
            nn.PReLU(3))
        if torch.cuda.device_count()>1:
            self.convlayer12 = nn.DataParallel(self.convlayer12)

    def add_noise(self,x):

        dtype = x.dtype
        device = x.device
        batch = x.size()[0]
        channel = x.size()[1]
        height = x.size()[2]
        width = x.size()[3]

        mean = torch.tensor(0,dtype = dtype)
        std = torch.tensor(0.005,dtype = dtype)
        size = (batch,1,height,width)

        noise_tensor = torch.normal(mean,std,size=size,device = device,dtype = dtype)
        noise_tensor = F.relu(noise_tensor+0.01)-0.01
        zero_padding = torch.zeros([batch,channel-1,height,width],device = device,dtype = dtype)
        noise_padd = torch.cat((noise_tensor,zero_padding),1)

        noise_x = torch.add(x,noise_padd)
        return noise_x
        
    def forward(self, x):
        x = self.add_noise(x)
        x = self.convlayer1(x)
        x = self.convlayer1_1(x)
        x = self.convlayer2(x)
        x = self.convlayer2_1(x)    
        x = self.convlayer3(x)
        x = self.convlayer3_1(x)
        x = self.convlayer4(x) + x
        x = self.convlayer5(x) + x
        x = self.convlayer6(x) + x
        x = self.convlayer7(x) + x
        x = self.convlayer8(x)
        x = self.convlayer9(x)
        x = self.convlayer9_1(x)
        x = self.convlayer10(x)
        x = self.convlayer10_1(x)
        x = self.convlayer11(x)
        x = self.convlayer12(x)
        return x


