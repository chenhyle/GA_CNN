#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import torch
import torch.utils.data as Data
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
from torchsummary import summary

import os
import random
import datetime
# from core_qnn.quaternion_layers import *
# from utils.QBN_Vecchi2 import QuaternionBatchNorm2d as QBatchNorm
from cliffordlayers.nn.modules.cliffordconv import CliffordConv2d
from cliffordlayers.nn.modules.test_clifford_convolution_li import CliffordConv2d_DZ

from cliffordlayers.nn.functional.batchnorm import (
    clifford_batch_norm,
    complex_batch_norm,
)
from cliffordlayers.nn.modules.batchnorm import (
    CliffordBatchNorm1d,
    CliffordBatchNorm2d,
    CliffordBatchNorm3d,
    ComplexBatchNorm1d,
)
from cliffordlayers.signature import CliffordSignature

from cliffordlayers.nn.modules.cliffordconv import (
    CliffordConv1d,
    CliffordConv2d,
    CliffordConv3d,
)
from cliffordlayers.nn.modules.cliffordlinear import CliffordLinear
import sys
sys.path.append(r'E:\py_practice\typhoon_predict')
from core_qnn.quaternion_layers import *
from utils.QBN_Vecchi2 import QuaternionBatchNorm2d as QBatchNorm
import time
from thop import profile
start_time1 = time.perf_counter()

# In[2]:


#  predition TI of leading time at 24 hours
pre_seq = 4
min_val_loss = 100
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# In[3]:


train = pd.read_csv('E:/py_practice/typhoon_predict/data/CMA_train_'+str(pre_seq*6)+'h.csv', header=None)
test= pd.read_csv('E:/py_practice/typhoon_predict/data/CMA_test_'+str(pre_seq*6)+'h.csv', header=None)
train.shape, test.shape


# In[4]:
CLIPER_feature = pd.concat((train, test), axis=0)
CLIPER_feature.reset_index(drop=True, inplace=True)


# In[5]:


X_wide_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()

X_wide = X_wide_scaler.fit_transform(CLIPER_feature.iloc[:, 5:])
X_wide_train = X_wide[0: train.shape[0], :]

y = y_scaler.fit_transform(CLIPER_feature.loc[:, 3].values.reshape(-1, 1))
y_train = y[0: train.shape[0], :]


# In[6]:


# now 6 hours ago  12 hours ago  18 hour ago
ahead_times = [0,1,2,3]

pressures = [1000, 750, 500, 250]

sequential_reanalysis_u_list = []
reanalysis_u_test_dict = {}
X_deep_u_scaler_dict = {}

sequential_reanalysis_v_list = []
reanalysis_v_test_dict = {}
X_deep_v_scaler_dict = {}


# In[7]:


reanalysis_type = 'u'
for ahead_time in ahead_times:
    reanalysis_list = []
    for pressure in pressures:
        folder = None
        if ahead_time == 0:
            folder = reanalysis_type
        else:
            folder = reanalysis_type + '_' + str(ahead_time*6)
            
        train_reanalysis_csv = pd.read_csv('E:/py_practice/typhoon_predict/data/ERA_Interim/'+folder+'/'+reanalysis_type+str(pressure)+'_train_31_31.csv', header=None)
        test_reanalysis_csv = pd.read_csv('E:/py_practice/typhoon_predict/data/ERA_Interim/'+folder+'/'+reanalysis_type+str(pressure)+'_test_31_31.csv', header=None)

        train_reanalysis = train_reanalysis_csv[train_reanalysis_csv[0].isin(train[0].unique())]
        test_reanalysis = test_reanalysis_csv[test_reanalysis_csv[0].isin(test[0].unique())]
        reanalysis_u_test_dict[reanalysis_type+str(pressure)+str(ahead_time)] = test_reanalysis # 保存test 用于后面测试
        
        reanalysis =  pd.concat((train_reanalysis, test_reanalysis), axis=0)
        reanalysis.reset_index(drop=True, inplace=True)

        scaler_name = reanalysis_type +str(pressure) + str(ahead_time)
        X_deep_u_scaler_dict[scaler_name] = MinMaxScaler()
        
        # 5:end is the 31*31 u component wind speed
        X_deep = X_deep_u_scaler_dict[scaler_name].fit_transform(reanalysis.loc[:, 5:])
        
        # (batch, type, channel, height, widht, time) here type is u
        X_deep_final = X_deep[0: train.shape[0], :].reshape(-1, 1, 1, 31, 31, 1)
        reanalysis_list.append(X_deep_final)

    X_deep_temp = np.concatenate(reanalysis_list[:], axis=2)
    print("ahead_time:", ahead_time, X_deep_temp.shape)
    sequential_reanalysis_u_list.append(X_deep_temp)

X_deep_u_train = np.concatenate(sequential_reanalysis_u_list, axis=5)


# In[8]:


reanalysis_type = 'v'
for ahead_time in ahead_times:
    reanalysis_list = []
    for pressure in pressures:
        folder = None
        if ahead_time == 0:
            folder = reanalysis_type
        else:
            folder = reanalysis_type + '_' + str(ahead_time*6)

        train_reanalysis_csv = pd.read_csv('E:/py_practice/typhoon_predict/data/ERA_Interim/'+folder+'/'+reanalysis_type+str(pressure)+'_train_31_31.csv', header=None)
        test_reanalysis_csv = pd.read_csv('E:/py_practice/typhoon_predict/data/ERA_Interim/'+folder+'/'+reanalysis_type+str(pressure)+'_test_31_31.csv', header=None)

        train_reanalysis = train_reanalysis_csv[train_reanalysis_csv[0].isin(train[0].unique())]
        test_reanalysis = test_reanalysis_csv[test_reanalysis_csv[0].isin(test[0].unique())]
        reanalysis_v_test_dict[reanalysis_type+str(pressure)+str(ahead_time)] = test_reanalysis # 保存test 用于后面测试

        reanalysis =  pd.concat((train_reanalysis, test_reanalysis), axis=0)
        reanalysis.reset_index(drop=True, inplace=True)

        scaler_name = reanalysis_type +str(pressure) + str(ahead_time)
        X_deep_v_scaler_dict[scaler_name] = MinMaxScaler()
        
        # 5:end is the 31*31 v component wind speed
        X_deep = X_deep_v_scaler_dict[scaler_name].fit_transform(reanalysis.loc[:, 5:])
        
        # (batch, type, channel, height, widht, time) here type is v
        X_deep_final = X_deep[0: train.shape[0], :].reshape(-1, 1, 1, 31, 31, 1)
        reanalysis_list.append(X_deep_final)
        
    X_deep_temp = np.concatenate(reanalysis_list[:], axis=2)
    print("ahead_time:", ahead_time, X_deep_temp.shape)
    sequential_reanalysis_v_list.append(X_deep_temp)

X_deep_v_train = np.concatenate(sequential_reanalysis_v_list, axis=5)


# In[10]:


X_deep_train = np.concatenate((X_deep_u_train, X_deep_v_train), axis=1)


# In[11]:


X_wide_train.shape, X_deep_train.shape


# In[12]:


class TrainLoader(Data.Dataset):
    def __init__(self, X_wide_train, X_deep_train, y_train):
        self.X_wide_train = X_wide_train
        self.X_deep_train = X_deep_train
        self.y_train = y_train
        
    def __getitem__(self, index):
        return [self.X_wide_train[index], self.X_deep_train[index]], self.y_train[index]
    
    def __len__(self):
        return len(self.X_wide_train)


# In[13]:
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # spatial attention
        self.clifford_att_block_1 = nn.Sequential(
            CliffordConv2d(g=[-1, -1], in_channels=32, out_channels=32, kernel_size=3, padding=1),
            CliffordBatchNorm2d([1, 1], channels=32),
            nn.ReLU(inplace=True),
            CliffordConv2d(g=[-1, -1], in_channels=32, out_channels=32, kernel_size=3, padding=1),
            CliffordBatchNorm2d([1, 1], channels=32),
            nn.Sigmoid(), )
        self.clifford_att_block_2 = nn.Sequential(
            CliffordConv2d(g=[-1, -1], in_channels=64, out_channels=64, kernel_size=3, padding=1),
            CliffordBatchNorm2d([1, 1], channels=64),
            nn.ReLU(inplace=True),
            CliffordConv2d(g=[-1, -1], in_channels=64, out_channels=64, kernel_size=3, padding=1),
            CliffordBatchNorm2d([1, 1], channels=64),
            nn.Sigmoid(), )
        self.clifford_att_block_3 = nn.Sequential(
            CliffordConv2d(g=[-1, -1], in_channels=128, out_channels=128, kernel_size=3, padding=1),
            CliffordBatchNorm2d([1, 1], channels=128),
            nn.ReLU(inplace=True),
            CliffordConv2d(g=[-1, -1], in_channels=128, out_channels=128, kernel_size=3, padding=1),
            CliffordBatchNorm2d([1, 1], channels=128),
            nn.Sigmoid(), )
        # self.cli_att_block_2 = nn.Sequential(
        #     CliffordConv2d_DZ(g=[-1], in_channels=128, out_channels=128, kernel_size=1, padding=0),
        #     CliffordBatchNorm2d([-1], channels=128),
        #     nn.ReLU(inplace=True),
        #     CliffordConv2d_DZ(g=[-1], in_channels=128, out_channels=128, kernel_size=1, padding=0),
        #     CliffordBatchNorm2d([-1], channels=128),
        #     nn.Sigmoid(),
        # )
        # self.cli_att_block_3 = nn.Sequential(
        #     CliffordConv2d_DZ(g=[-1], in_channels=256, out_channels=256, kernel_size=1, padding=0),
        #     CliffordBatchNorm2d([-1], channels=256),
        #     nn.ReLU(inplace=True),
        #     CliffordConv2d_DZ(g=[-1], in_channels=256, out_channels=256, kernel_size=1, padding=0),
        #     CliffordBatchNorm2d([-1], channels=256),
        #     nn.Sigmoid(),
        # )
        #
        # self.att_block_2 = nn.Sequential(
        #     nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, padding=0),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, padding=0),
        #     nn.BatchNorm2d(128),
        #     nn.Sigmoid(),
        # )
        # self.att_block_3 = nn.Sequential(
        #     nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, padding=0),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, padding=0),
        #     nn.BatchNorm2d(256),
        #     nn.Sigmoid(),
        # )
        # cross
        self.cross_unit = nn.Parameter(data=torch.ones(len(ahead_times), 6))
        # fuse
        self.fuse_unit = nn.Parameter(data=torch.ones(len(ahead_times), 4))

        self.pool = nn.MaxPool3d(kernel_size=(2, 2, 1), stride=(2, 2, 1))
        self.clifford_conv1 = CliffordConv2d(g=[-1, -1], in_channels=2, out_channels=32, kernel_size=3, padding=1)
        self.clifford_conv2 = CliffordConv2d(g=[-1, -1], in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.clifford_conv3 = CliffordConv2d(g=[-1, -1], in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.cli_linear=CliffordLinear(g=[-1, -1], in_channels=128 * 3 * 3, out_channels= 128)
        # self.fc1 = nn.Linear(128 * 3 * 3, 128)
        self.fc2 = nn.Linear(96 + 128 * len(ahead_times), 64)
        self.fc3 = nn.Linear(64, 1)
        # self.fc4 = nn.Linear(2,1)


    def forward(self, wide, deep):
        seq_list = []
        for i in range(len(ahead_times)):

            # 1、对uv层进行卷积
            deep_uv = deep[:, :, :, :, :, i]
            deep_uv = deep_uv.permute(0, 1, 3, 4, 2)  # 把压强换到最后[128,2,31,31,4]
            # 2、更改split 1
            deep_uv = self.pool(F.relu(self.clifford_conv1(deep_uv)))  #[128,32,15,15,4]
            deep_uv =self.clifford_att_block_1(deep_uv)*deep_uv#[128,32,15,15,4]
            # fuse 1
            #time_seq_1=[128,64,7,7,4]
            time_seq_1 = self.pool(F.relu(self.clifford_conv2(deep_uv)))
            #time_seq_1 =[128,64,7,7,4]
            time_seq_1 = self.clifford_att_block_2(time_seq_1) * time_seq_1

            # 更改split 2
            #deep_uv =[128,64,7,7,4]
            deep_uv = self.pool(F.relu(self.clifford_conv2(deep_uv)))
            #deep_uv =[128,64,7,7,4]
            deep_uv = self.clifford_att_block_2(deep_uv)*deep_uv

            # print("self.clifford_att_block_1层参数", float(count_params(self.clifford_att_block_1)))
            # print("self.clifford_att_block_2层参数", float(count_params(self.clifford_att_block_2)))
            # print("self.clifford_att_block_3层参数", float(count_params(self.clifford_att_block_3)))
            # print("clifford_conv1 层参数", float(count_params(self.clifford_conv1)))
            # print("clifford_conv2 层参数", float(count_params(self.clifford_conv2)))
            # print("clifford_conv3层参数", float(count_params(self.clifford_conv3)))
            # print("cli_linear层参数", float(count_params(self.cli_linear)))
            # print("fc2层参数", float(count_params(self.fc2)))eq_2 = self.fuse_unit[i][0] / (self.fuse_unit[i][0] + self.fuse_unit[i][1]) * time_seq_1 + \
            #                          self.fuse_unit[i][1] / (self.fuse_unit[i][0] + self.fuse_unit[i][1]) * deep_uv
            # print("fc3层参数", float(count_params(self.fc3)))
            # fuse 2
            time_s
            #deep_uv =[128,128,3,3,4]
            time_seq_2 = self.pool(F.relu(self.clifford_conv3(time_seq_2)))
            #time_seq_2 =[128,128,3,3,4]
            time_seq_2 = self.clifford_att_block_3(time_seq_2) * time_seq_2

            # 4、更改split 3
            #deep_uv =[128,128,3,3,4]
            deep_uv = self.pool(F.relu(self.clifford_conv3(deep_uv)))
            #deep_uv =[128,128,3,3,4]
            deep_uv = self.clifford_att_block_3(deep_uv) * deep_uv

            # fuse 3
            time_seq = self.fuse_unit[i][2] / (self.fuse_unit[i][2] + self.fuse_unit[i][3]) * time_seq_2 + \
                       self.fuse_unit[i][3] / (self.fuse_unit[i][2] + self.fuse_unit[i][3]) * deep_uv
            #time_seq =[128,128,3,3,4]
            time_seq = self.clifford_att_block_3(time_seq) * time_seq
            #time_seq=[128,128*3*3,4]
            time_seq = time_seq.view(-1, 128 * 3 * 3,4)
            #time_seq=[128,128,4]
            time_seq = self.cli_linear(time_seq)
            t1=time_seq[:,:,0]
            t2=time_seq[:,:,1]
            t3=time_seq[:,:,2]
            t4=time_seq[:,:,3]
            time_seq = (self.fuse_unit[i][0] / (self.fuse_unit[i][0] + self.fuse_unit[i][1]+self.fuse_unit[i][2] + self.fuse_unit[i][3])) * t1 +(self.fuse_unit[i][1] / (self.fuse_unit[i][0] + self.fuse_unit[i][1]+self.fuse_unit[i][2] + self.fuse_unit[i][3])) * t2+ \
                       (self.fuse_unit[i][2] / (self.fuse_unit[i][0] + self.fuse_unit[i][1]+self.fuse_unit[i][2] + self.fuse_unit[i][3])) * t3 + \
                       (self.fuse_unit[i][3] / (self.fuse_unit[i][0] + self.fuse_unit[i][1]+self.fuse_unit[i][2] + self.fuse_unit[i][3]))* t4

            seq_list.append(time_seq)
        wide = wide.view(-1, 96)
        wide_n_deep = torch.cat((wide, seq_list[0]), 1)
        if len(ahead_times) > 1:
            for i in range(1, len(ahead_times)):
                wide_n_deep = torch.cat((wide_n_deep, seq_list[i]), 1)
        wide_n_deep = F.relu(self.fc2(wide_n_deep))
        wide_n_deep = F.relu(self.fc3(wide_n_deep))
        return wide_n_deep


net = Net()


# In[14]:


#Xavier 初始化
import torch.nn.init as init
def xavier_init(m):
    if isinstance(m,(nn.Linear,nn.Conv2d)):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias,0)

def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count
print("Total trainable parameters:", count_param(net))
from torchinfo import summary
summary(net)
# In[15]:


#重新实例化模型
net = Net()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = net.to(device)
net.apply(xavier_init)


# In[16]:


batch_size = 128
epochs = 96
lr = 0.0004
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
model_name = 'TIP_cnn_4(1).pkl'


# In[17]:


import logging
#清除已经存在的日志处理器
logging.getLogger().handlers = []
log_name = 'TIP_cnn_4(1)'
logging.basicConfig(filename=log_name,level=logging.INFO)
logging.info(f"原模型 数据集拿到外面分")
logging.info(f"epochs:{epochs};lr:{lr};batch_size={batch_size}")
logging.info(f"损失函数:L1Loss();优化器:Adam")


# In[18]:


full_train_index = [*range(0, len(X_wide_train))]
train_index, val_index, _, _, = train_test_split(full_train_index,full_train_index,test_size=0.2)
train_dataset = torch.utils.data.DataLoader(
    TrainLoader(X_wide_train[train_index], X_deep_train[train_index], y_train[train_index]), 
                                                 batch_size=batch_size, shuffle=True)
val_dataset = torch.utils.data.DataLoader(
    TrainLoader(X_wide_train[val_index], X_deep_train[val_index], y_train[val_index]), 
                                                 batch_size=batch_size, shuffle=True)

# for epoch in range(epochs):  # loop over the dataset multiple times
#     start_time2 = time.perf_counter()
#     # training
#     total_train_loss = 0
#     train_count = 0
#     net.train()
#     for step, (batch_x, batch_y) in enumerate(train_dataset):
#         if torch.cuda.is_available():
#             net.cuda()
#             X_wide_train_cuda = batch_x[0].float().cuda()
#             X_deep_train_cuda = batch_x[1].float().cuda()
#             y_train_cuda = batch_y.float().cuda()
#         # zero the parameter gradients
#         optimizer.zero_grad()
#
#         # forward + backward + optimize
#         pred_y = net(X_wide_train_cuda, X_deep_train_cuda)
#         loss = criterion(pred_y, y_train_cuda)
#         total_train_loss += loss.item()
#         train_count +=1
#         loss.backward()
#         optimizer.step()
#
#     # validating
#     total_val_loss = 0
#     val_count = 0
#     net.eval()
#     with torch.no_grad():
#         for _,(batch_val_x, batch_val_y) in enumerate(val_dataset):
#
#             if torch.cuda.is_available():
#                 X_wide_val_cuda = batch_val_x[0].float().cuda()
#                 X_deep_val_cuda = batch_val_x[1].float().cuda()
#                 y_val_cuda = batch_val_y.cuda()
#
#             pred_y = net(X_wide_val_cuda, X_deep_val_cuda)
#             val_loss = criterion(pred_y, y_val_cuda)
#             total_val_loss += val_loss.item()
#             val_count += 1
#             # print statistics
#         if min_val_loss > total_val_loss/val_count:
#             torch.save(net.state_dict(), model_name)
#             min_val_loss = total_val_loss
#         end_time2 = time.perf_counter()
#         run_time2 = end_time2 - start_time2
#         print("迭代一次时间", run_time2, "秒")
#         print('epochs [%d/%d] train_loss: %.5f val_loss: %.5f' % (epoch + 1, epochs, total_train_loss/train_count, total_val_loss/val_count))
#         logging.info(f"Epoch {epoch+1}/{epochs}; train_loss:{total_train_loss/train_count:.5f};val_loss:{total_val_loss/val_count:.5f}")

print('Finished Training')


# In[19]:


net.load_state_dict(torch.load(model_name))

years = test[4].unique()

test_list = []

for year in years:
    temp = test[test[4]==year]
    temp = temp.reset_index(drop=True)
    test_list.append(temp)
    
len(test_list)


# In[21]:


with torch.no_grad():
    for year, _test in zip(years, test_list):

        print(year, '年:')
        y_test = _test.loc[:,3]
        y_test_scaler = y_scaler.transform(_test.loc[:,3].values.reshape(-1, 1))
    
        X_wide_test = X_wide_scaler.transform(_test.loc[:,5:])

        final_test_u_list = []
        for ahead_time in ahead_times:
            year_test_list = []
            for pressure in pressures:
                scaler_name = 'u' +str(pressure) + str(ahead_time)
                X_deep = reanalysis_u_test_dict[scaler_name][reanalysis_u_test_dict[scaler_name][0].isin(_test[0].unique())].loc[:,5:]
                X_deep = X_deep_u_scaler_dict[scaler_name].transform(X_deep)
                X_deep_final = X_deep.reshape(-1, 1, 1, 31, 31, 1)
                year_test_list.append(X_deep_final)
            X_deep_temp = np.concatenate(year_test_list, axis=2)
            final_test_u_list.append(X_deep_temp)
        X_deep_u_test = np.concatenate(final_test_u_list, axis=5)
        
        final_test_v_list = []
        for ahead_time in ahead_times:
            year_test_list = []
            for pressure in pressures:
                scaler_name = 'v' +str(pressure) + str(ahead_time)
                X_deep = reanalysis_v_test_dict[scaler_name][reanalysis_v_test_dict[scaler_name][0].isin(_test[0].unique())].loc[:,5:]
                X_deep = X_deep_v_scaler_dict[scaler_name].transform(X_deep)
                X_deep_final = X_deep.reshape(-1, 1, 1, 31, 31, 1)
                year_test_list.append(X_deep_final)
            X_deep_temp = np.concatenate(year_test_list, axis=2)
            final_test_v_list.append(X_deep_temp)
        X_deep_v_test = np.concatenate(final_test_v_list, axis=5)

    
        X_deep_test = np.concatenate((X_deep_u_test, X_deep_v_test), axis=1)
        
        if torch.cuda.is_available():
            X_wide_test = torch.from_numpy(X_wide_test).float().cuda()
            X_deep_test = torch.from_numpy(X_deep_test).float().cuda()
            y_test_scaler = torch.from_numpy(y_test_scaler).float().cuda()
        
        #预测函数损失  
        pred = net(X_wide_test, X_deep_test)
        pred_scaler = pred
        loss_scaler = criterion(pred, y_test_scaler)
        print('loss_scaler:', loss_scaler.item())

        # 计算预测过程中的FLOPs和参数数量
        flops, params = profile(net, inputs=(X_wide_test, X_deep_test))
        print('flops: ', flops, 'params: ', params)
        print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))
        # 计算未归一化损失
        pred = y_scaler.inverse_transform(pred.cpu().detach().numpy().reshape(-1, 1))
        true = y_test.values.reshape(-1, 1)
        diff = np.abs(pred - true)

        mae = sum(diff) / len(diff)
        print('mae wind error:', mae)

        rmse = np.sqrt(sum(diff ** 2) / len(diff))
        print('rmse wind error:', rmse)
        print('avg wind error:', sum(diff) / len(diff))

        # 计算RAE
        rae = sum(diff) / sum(np.abs(true - np.mean(true)))
        print('RAE wind error:', rae)

        # 计算RSE
        rse = np.sqrt(sum(diff ** 2) / sum((true - np.mean(true)) ** 2))
        print('RSE wind error:', rse)

        # 计算预测值和真实值之间的相关系数
        corr = np.corrcoef(pred.flatten(), true.flatten())[0, 1]
        print('Correlation coefficient:', corr)
        
        logging.info(f" Year {year};loss_scaler:{loss_scaler.item():.5f}; mae wind error:{mae}; rmse wind error:{rmse}")
  
logging.info(f"FINISH")
logging.shutdown()


# In[ ]:

end_time1 = time.perf_counter()
run_time1 = (end_time1 - start_time1)/60
print("运行时间",run_time1,"分钟")



# In[ ]:





# In[ ]:




