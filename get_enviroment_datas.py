from typing import List, Callable, Union, Any, TypeVar, Tuple
from torch import nn
import torch
from torch.nn import functional as F
import numpy as np
import pandas as pd
from torch import tensor as Tensor
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import random
from models.rnn import CMV_LSTM, MV_LSTM
from models.vae import ConditionalVAE, VanillaVAEEncoder, VanillaVAEDecoder
from models.rvae import UnitRVAE, RVAE
from utils.loss import VAE_Loss
from pretrain_rvae_model import get_total_datas, timeseries


seq_len = 5
batch_size = 4
torch.autograd.set_detect_anomaly(True)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load datas
df_econs = pd.read_csv('data/total_economy_data.csv', index_col=0)
df_econs = df_econs[df_econs['year'] > 2015].reset_index(drop=True, inplace=False)
df_econs1 = df_econs.drop(columns=['year', 'month'])
df_econs1 = df_econs1.dropna(axis=1)
arr_econs = df_econs1.to_numpy()
arr_econs = arr_econs / 1e10

df_trends = pd.read_csv('data/total_trends.csv', index_col=0)
df_trends = df_trends[df_trends['year'] > 2015].reset_index(drop=True, inplace=False)
df_trends1 = df_trends.drop(columns=['year', 'month'])
df_trends1 = df_trends1.dropna(axis=1)
arr_trends = df_trends1.to_numpy() / 20

df_weathers = pd.read_csv('data/average_weather.csv', index_col=0)
df_weathers = df_weathers[df_weathers['year'] > 2015].reset_index(drop=True, inplace=False)
df_weathers1 = df_weathers.drop(columns=['year', 'month'])
df_weathers1 = df_weathers1.dropna(axis=1)
arr_weathers = df_weathers1.to_numpy() / 10

# Train RVAE
rvae = RVAE(
    arr_econs.shape[-1],
    arr_trends.shape[-1],
    arr_weathers.shape[-1],
    28,
    16,
    4,
    seq_len
    )
rvae.rnn.init_hidden(batch_size)
rvae.load_state_dict(torch.load('rvae.pt'))

model = rvae
num_points = len(arr_econs)
base_num = int(num_points * 0.8)

x_train1, y_train1 = get_total_datas(seq_len, arr_econs, shuffle=False)
train_dataset1 = timeseries(x_train1, y_train1)

x_train2, y_train2 = get_total_datas(seq_len, arr_trends, shuffle=False)
train_dataset2 = timeseries(x_train2, y_train2)

x_train3, y_train3 = get_total_datas(seq_len, arr_weathers, shuffle=False)
train_dataset3 = timeseries(x_train3, y_train3)

model = rvae.to(device)

train_train_loader1 = DataLoader(train_dataset1, shuffle=False, batch_size=batch_size)
train_train_loader2 = DataLoader(train_dataset2, shuffle=False, batch_size=batch_size)
train_train_loader3 = DataLoader(train_dataset3, shuffle=False, batch_size=batch_size)
for data1, data2, data3 in zip(train_train_loader1, train_train_loader2, train_train_loader3):
    if data1[0].shape[0] != batch_size:
        break
    curr_data1 = data1[0].clone().detach(), data1[1].clone().detach()
    curr_data2 = data2[0].clone().detach(), data2[1].clone().detach()
    curr_data3 = data3[0].clone().detach(), data3[1].clone().detach()
    output1 = model.vae_encoder1(curr_data1[0])[0]
    output2 = model.vae_encoder2(curr_data1[0])[0]
    output3 = model.vae_encoder3(curr_data1[0])[0]
    output = torch.vstack([output1, output2, output3], axis=2)
    #loss = criterion(output[2], output[3], output[0], curr_data[1])
    result = model.loss_function([curr_data1[1].clone(), curr_data2[1].clone(), curr_data3[1].clone()], *output, **{'M_N': 1})

