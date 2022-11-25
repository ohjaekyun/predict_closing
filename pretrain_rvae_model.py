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

# VAE와 LSTM을 섞인 모델 생성 후 저장
seq_len = 5
batch_size = 4
torch.autograd.set_detect_anomaly(True)
device = "cuda" if torch.cuda.is_available() else "cpu"


def get_total_datas(seq_length, data, shuffle=True):
    total_num = len(data)
    list_x_datas = []
    list_y_datas = []
    for i in range(seq_length, total_num):
        list_y_datas.append(data[i].reshape([1, *data[i].shape]))
        list_x_datas.append(data[i - seq_length:i].copy().reshape([1, *data[i - seq_length:i].shape]))
    
    if shuffle:
        suffled_idices = list(range(len(list_x_datas)))
        random.shuffle(suffled_idices)
        list_x_datas1 = []
        list_y_datas1 = []
        for i in suffled_idices:
            list_x_datas1.append(list_x_datas[i])
            list_y_datas1.append(list_y_datas[i])
        return np.vstack(list_x_datas1), np.vstack(list_y_datas1)
    return np.vstack(list_x_datas), np.vstack(list_y_datas)


class timeseries(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.len = x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
  
    def __len__(self):
        return self.len


if __name__ == '__main__':
    df_econs = pd.read_csv('data/total_economy_data.csv', index_col=0)
    df_econs1 = df_econs.drop(columns=['year', 'month'])
    df_econs1 = df_econs1.dropna(axis=1)
    arr_econs = df_econs1.to_numpy()
    arr_econs = arr_econs / 1e10

    # Pretrain
    train_data = arr_econs
    urvae = UnitRVAE(arr_econs.shape[-1], seq_len, 28)

    x_train, y_train = get_total_datas(seq_len, train_data, shuffle=True)
    ts_dataset = timeseries(x_train, y_train)


    model = urvae.to(device)
    urvae.rnn.init_hidden(batch_size)
    criterion = VAE_Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    epochs = 50

    for i in range(epochs):
        ts_train_loader = DataLoader(ts_dataset, shuffle=True, batch_size=batch_size)
        model.train()
        for j, data in enumerate(ts_train_loader):
            if data[0].shape[0] != batch_size:
                break
            curr_data = data[0].clone().detach(), data[1].clone().detach()
            output = model(curr_data[0])
            loss = criterion(output[2], output[3], output[0], curr_data[1])

            optimizer.zero_grad()
            loss.backward(retain_graph=False)
            optimizer.step()
    
        if i % 2 == 0:
            print(i, 'th iteration : ', loss)


    # Load datas
    df_econs = pd.read_csv('numble/predict_closing/data/total_economy_data.csv', index_col=0)
    df_econs = df_econs[df_econs['year'] > 2015].reset_index(drop=True, inplace=False)
    df_econs1 = df_econs.drop(columns=['year', 'month'])
    df_econs1 = df_econs1.dropna(axis=1)
    arr_econs = df_econs1.to_numpy()
    arr_econs = arr_econs / 1e10

    df_trends = pd.read_csv('numble/predict_closing/data/total_trends.csv', index_col=0)
    df_trends = df_trends[df_trends['year'] > 2015].reset_index(drop=True, inplace=False)
    df_trends1 = df_trends.drop(columns=['year', 'month'])
    df_trends1 = df_trends1.dropna(axis=1)
    arr_trends = df_trends1.to_numpy() / 20

    df_weathers = pd.read_csv('numble/predict_closing/data/average_weather.csv', index_col=0)
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
    rvae.vae_encoder1 = model.vae_encoder
    rvae.vae_decoder1 = model.vae_decoder
    num_points = len(arr_econs)
    base_num = int(num_points * 0.8)

    train_data1 = arr_econs[:base_num]
    test_data1 = arr_econs[base_num:]
    train_data2 = arr_trends[:base_num]
    test_data2 = arr_trends[base_num:]
    train_data3 = arr_weathers[:base_num]
    test_data3 = arr_weathers[base_num:]

    x_train1, y_train1 = get_total_datas(seq_len, train_data1, shuffle=True)
    train_dataset1 = timeseries(x_train1, y_train1)
    x_test1, y_test1 = get_total_datas(seq_len, test_data1, shuffle=True)
    test_dataset1 = timeseries(x_test1, y_test1)

    x_train2, y_train2 = get_total_datas(seq_len, train_data2, shuffle=True)
    train_dataset2 = timeseries(x_train2, y_train2)
    x_test2, y_test2 = get_total_datas(seq_len, test_data2, shuffle=True)
    test_dataset2 = timeseries(x_test2, y_test2)

    x_train3, y_train3 = get_total_datas(seq_len, train_data3, shuffle=True)
    train_dataset3 = timeseries(x_train3, y_train3)
    x_test3, y_test3 = get_total_datas(seq_len, test_data3, shuffle=True)
    test_dataset3 = timeseries(x_test3, y_test3)

    model = rvae.to(device)
    rvae.rnn.init_hidden(batch_size)
    criterion = VAE_Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    epochs = 50

    for i in range(epochs):
        train_train_loader1 = DataLoader(train_dataset1, shuffle=True, batch_size=batch_size)
        test_train_loader1 = DataLoader(test_dataset1, shuffle=True, batch_size=batch_size)
        train_train_loader2 = DataLoader(train_dataset2, shuffle=True, batch_size=batch_size)
        test_train_loader2 = DataLoader(test_dataset2, shuffle=True, batch_size=batch_size)
        train_train_loader3 = DataLoader(train_dataset3, shuffle=True, batch_size=batch_size)
        test_train_loader3 = DataLoader(test_dataset3, shuffle=True, batch_size=batch_size)
        model.train()
        for data1, data2, data3 in zip(train_train_loader1, train_train_loader2, train_train_loader3):
            if data1[0].shape[0] != batch_size:
                break
            curr_data1 = data1[0].clone().detach(), data1[1].clone().detach()
            curr_data2 = data2[0].clone().detach(), data2[1].clone().detach()
            curr_data3 = data3[0].clone().detach(), data3[1].clone().detach()
            output = model(curr_data1[0], curr_data2[0], curr_data3[0])
            #loss = criterion(output[2], output[3], output[0], curr_data[1])
            result = model.loss_function([curr_data1[1].clone(), curr_data2[1].clone(), curr_data3[1].clone()], *output, **{'M_N': 0.5})
            loss = result['loss']
            optimizer.zero_grad()
            loss.backward(retain_graph=False)
            optimizer.step()
        model.eval()
        for data1, data2, data3 in zip(test_train_loader1, test_train_loader2, test_train_loader3):
            if data[0].shape[0] != batch_size:
                break
            curr_data1 = data1[0].clone().detach(), data1[1].clone().detach()
            curr_data2 = data2[0].clone().detach(), data2[1].clone().detach()
            curr_data3 = data3[0].clone().detach(), data3[1].clone().detach()
            output = model(curr_data1[0], curr_data2[0], curr_data3[0])
            #val_loss = criterion(output[2], output[3], output[0], curr_data[1])
            result = model.loss_function([curr_data1[1].clone(), curr_data2[1].clone(), curr_data3[1].clone()], *output, **{'M_N': 0.5})
            val_loss = result['loss']
        if i % 2 == 0:
            print(i, 'th iteration : ', val_loss)
