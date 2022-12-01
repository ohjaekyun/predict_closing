from typing import List, Callable, Union, Any, TypeVar, Tuple
from torch import nn
import torch
from torch.nn import functional as F
import numpy as np
import pandas as pd
from torch import tensor as Tensor
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.model_selection import train_test_split
import random
from models.rnn import CMV_LSTM, MV_LSTM
from models.vae import ConditionalVAE, VanillaVAEEncoder, VanillaVAEDecoder
from models.rvae import UnitRVAE, RVAE
from utils.loss import VAE_Loss


# VAE와 LSTM을 섞인 모델 생성 후 저장
seq_len = 10
batch_size = 5
test_batch_size = 3
seed = 42
torch.autograd.set_detect_anomaly(True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_total_datas(seq_length, data, shuffle=True, seed=42):
    total_num = len(data)
    list_x_datas = []
    list_y_datas = []
    for i in range(seq_length, total_num):
        for j in range(seq_length, i + 1):
            data_y = torch.tensor(data[i], dtype=torch.float32)
            data_x = torch.tensor(data[i - j:i], dtype=torch.float32)
            list_y_datas.append(data_y.reshape([1, *data[i].shape]))
            list_x_datas.append(data_x.reshape([1, *data[i - j:i].shape]))
    
    if shuffle:
        list_x_datas1, list_y_datas1 = shuffle_data_list(list_x_datas, list_y_datas, seed=seed)
        return (list_x_datas1), (list_y_datas1)
    return (list_x_datas), (list_y_datas)


def get_total_datas_by_length(seq_length, data, shuffle=True, seed=42, min_batch_size=1):
    total_num = len(data)
    dict_x_datas = {}
    dict_y_datas = {}
    for j in range(seq_length, total_num - min_batch_size + 1):
        dict_x_datas[j] = []
        dict_y_datas[j] = []
        for i in range(total_num - j):
            data_y = torch.tensor(data[i + j], dtype=torch.float32)
            data_x = torch.tensor(data[i:i + j], dtype=torch.float32)
            dict_y_datas[j].append(data_y.reshape([1, *data[i + j].shape]))
            dict_x_datas[j].append(data_x.reshape([1, *data[i:i + j].shape]))

    if shuffle:
        random.seed(seed)
        dict_x_datas1 = {}
        dict_y_datas1 = {}
        for j in range(seq_length, total_num - min_batch_size + 1):
            #set_seeds(random.randint(1, 512))
            list_x_datas, list_y_datas = shuffle_data_list(dict_x_datas[j], dict_y_datas[j], seed=random.randint(1, 512))
            dict_x_datas1[j] = list_x_datas
            dict_y_datas1[j] = list_y_datas
        return (dict_x_datas1), (dict_y_datas1)
    return (dict_x_datas), (dict_y_datas)


def get_test_list(seq_length, data, base_idx):
    test_list = []
    total_num = len(data)
    for i in range(base_idx, total_num):
        for j in range(seq_length, i + 1):
            test_list.append((torch.tensor(data[i - j:i], dtype=torch.float32), torch.tensor(data[i], dtype=torch.float32)))
    return test_list


def get_test_datas_by_length(seq_length, data, base_idx, min_batch_size=1):
    total_num = len(data)
    dict_x_datas = {}
    dict_y_datas = {}
    for j in range(seq_length, total_num - min_batch_size + 1):
        list_xs = []
        list_ys = []
        for i in range(total_num - j):
            if i + j >= base_idx:
                data_y = torch.tensor(data[i + j], dtype=torch.float32)
                data_x = torch.tensor(data[i:i + j], dtype=torch.float32)
                list_ys.append(data_y.reshape([1, *data[i + j].shape]))
                list_xs.append(data_x.reshape([1, *data[i:i + j].shape]))
        if len(list_xs) > 0:
            dict_x_datas[j] = list_xs
            dict_y_datas[j] = list_ys
    return (dict_x_datas), (dict_y_datas)


def shuffle_data_list(*args, seed=42):
    random.seed(seed)
    new_lists = []
    shuffled_indices = list(range(len(args[0])))
    random.shuffle(shuffled_indices)
    for arg in args:
        datas = []
        for idx in shuffled_indices:
            datas.append(arg[idx])
        new_lists.append(datas)
    return new_lists


class timeseries(Dataset):
    def __init__(self, x, y):
        self.x = x # torch.tensor(x, dtype=torch.float32)
        self.y = y # torch.tensor(y, dtype=torch.float32)
        self.len = len(x) # x.shape[0]

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
    base_idx = int(len(arr_econs) * 0.8)

    # Pretrain
    train_data = arr_econs[:base_idx]
    urvae = UnitRVAE(n_features=arr_econs.shape[-1], seq_length=seq_len, latent_dim=28)

    #x_train, y_train = get_total_datas(seq_len, train_data, shuffle=True)
    d_x_test, d_y_test = get_test_datas_by_length(seq_len * 2, arr_econs, base_idx, min_batch_size=3)
    d_x_train, d_y_train = get_total_datas_by_length(seq_len, train_data, shuffle=True, min_batch_size=2 * batch_size)

    model = urvae.to(device)
    urvae.rnn.init_hidden()
    criterion = VAE_Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, 10, 1)
    epochs = 1

    set_seeds(seed)
    key_list = list(d_x_train.keys())
    test_key_list = list(d_x_test.keys())

    test_loss = 500
    early_stopping_rounds = 60
    tolerance = 0.1
    threshold = early_stopping_rounds
    for i in range(epochs):
        all_loss = steps = 0
        model.train()
        for step in range(5):
            base_length = random.choice(key_list)
            ts_dataset = timeseries(d_x_train[base_length], d_y_train[base_length])
            ts_train_loader = DataLoader(ts_dataset, shuffle=True, batch_size=batch_size)
            iters = len(ts_train_loader)
            
            for j, data in enumerate(ts_train_loader):
                if data[0].shape[0] != batch_size:
                    j = max(0, j - 1)
                    break
                curr_data = data[0].clone().detach(), data[1].clone().detach()
                output = model(curr_data[0].squeeze(1))
                loss = criterion(output[2], output[3], output[0], curr_data[1])
                all_loss += loss.item()
                optimizer.zero_grad()
                loss.backward(retain_graph=False)
                optimizer.step()
                scheduler.step(i + j / iters)
            steps += (j + 1)
        if i % 2 == 0:
            print(f'{i}-th iteration TRAIN LOSS: ', all_loss / (steps))

        model.eval()
        all_loss = steps = 0 
        for step in range(5):
            base_length = random.choice(test_key_list)
            ts_dataset = timeseries(d_x_test[base_length], d_y_test[base_length])
            ts_test_loader = DataLoader(ts_dataset, shuffle=True, batch_size=test_batch_size)
            for j, data in enumerate(ts_test_loader):
                if data[0].shape[0] != test_batch_size:
                    j = max(0, j - 1)
                    break
                curr_data = data[0].clone().detach(), data[1].clone().detach()
                output = model(curr_data[0].squeeze(1))
                loss = criterion(output[2], output[3], output[0], curr_data[1])
                all_loss += loss.item()
            steps += (j + 1)
        curr_loss = all_loss / steps
        if curr_loss < test_loss:
            test_loss = curr_loss
            #torch.save(model.state_dict(), 'best_econ_time_series1.pt')
            threshold = early_stopping_rounds
        elif curr_loss > test_loss * (1 + tolerance):
            threshold -= 1
            if threshold < 0:
                break
        if i % 2 == 0:
            print(f'{i}-th iteration TEST LOSS: ', curr_loss)

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
    rvae.vae_encoder1 = model.vae_encoder
    rvae.vae_decoder1 = model.vae_decoder
    num_points = len(arr_econs)
    base_num = int(num_points * 0.8)

    train_data1 = arr_econs[:base_num]
    d_x_test1, d_y_test1 = get_test_datas_by_length(seq_len * 2, arr_econs, base_num, min_batch_size=3)
    d_x_train1, d_y_train1 = get_total_datas_by_length(seq_len, train_data1, shuffle=True, min_batch_size=2 * batch_size)
    train_data1 = arr_econs[:base_num]
    test_data1 = arr_econs[base_num:]

    train_data2 = arr_trends[:base_num]
    test_data2 = arr_trends[base_num:]
    d_x_test2, d_y_test2 = get_test_datas_by_length(seq_len * 2, arr_trends, base_num, min_batch_size=3)
    d_x_train2, d_y_train2 = get_total_datas_by_length(seq_len, train_data2, shuffle=True, min_batch_size=2 * batch_size)

    train_data3 = arr_weathers[:base_num]
    test_data3 = arr_weathers[base_num:]
    d_x_test3, d_y_test3 = get_test_datas_by_length(seq_len * 2, arr_weathers, base_num, min_batch_size=3)
    d_x_train3, d_y_train3 = get_total_datas_by_length(seq_len, train_data3, shuffle=True, min_batch_size=2 * batch_size)

    rvae.rnn.init_hidden()
    rvae.rnn.load_single_hidden(model.rnn)
    model = rvae.to(device)
    
    criterion = VAE_Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, 10, 1)
    epochs = 120
    set_seeds(seed)
    key_list = list(d_x_train1.keys())
    test_key_list = list(d_x_test1.keys())

    test_loss = 500
    early_stopping_rounds = 20
    tolerance = 0.1
    threshold = early_stopping_rounds

    for i in range(epochs):
        model.train()
        for step in range(5):
            base_length = random.choice(key_list)
            ts_dataset1 = timeseries(d_x_train1[base_length], d_y_train1[base_length])
            ts_train_loader1 = DataLoader(ts_dataset1, shuffle=True, batch_size=batch_size)
            iters = len(ts_train_loader1)
            ts_dataset2 = timeseries(d_x_train2[base_length], d_y_train2[base_length])
            ts_train_loader2 = DataLoader(ts_dataset2, shuffle=True, batch_size=batch_size)
            ts_dataset3 = timeseries(d_x_train3[base_length], d_y_train3[base_length])
            ts_train_loader3 = DataLoader(ts_dataset3, shuffle=True, batch_size=batch_size)
            for j, (data1, data2, data3) in enumerate(zip(ts_train_loader1, ts_train_loader2, ts_train_loader3)):
                if data1[0].shape[0] != batch_size:
                    break
                curr_data1 = data1[0].clone().detach(), data1[1].clone().detach()
                curr_data2 = data2[0].clone().detach(), data2[1].clone().detach()
                curr_data3 = data3[0].clone().detach(), data3[1].clone().detach()
                output = model(curr_data1[0].squeeze(1), curr_data2[0].squeeze(1), curr_data3[0].squeeze(1))
                #loss = criterion(output[2], output[3], output[0], curr_data[1])
                result = model.loss_function([curr_data1[1].squeeze(1).clone(), curr_data2[1].squeeze(1).clone(), curr_data3[1].squeeze(1).clone()], *output, **{'M_N': 1})
                loss = result['loss']
                optimizer.zero_grad()
                loss.backward(retain_graph=False)
                optimizer.step()
                scheduler.step(i + j / iters)
        model.eval()
        all_loss = steps = 0 
        for step in range(5):
            base_length = random.choice(test_key_list)
            ts_dataset1 = timeseries(d_x_test1[base_length], d_y_test1[base_length])
            ts_test_loader1 = DataLoader(ts_dataset1, shuffle=True, batch_size=test_batch_size)
            ts_dataset2 = timeseries(d_x_test2[base_length], d_y_test2[base_length])
            ts_test_loader2 = DataLoader(ts_dataset2, shuffle=True, batch_size=test_batch_size)
            ts_dataset3 = timeseries(d_x_test3[base_length], d_y_test3[base_length])
            ts_test_loader3 = DataLoader(ts_dataset3, shuffle=True, batch_size=test_batch_size)
            for j, (data1, data2, data3) in enumerate(zip(ts_test_loader1, ts_test_loader2, ts_test_loader3)):
                if data1[0].shape[0] != test_batch_size:
                    j = max(0, j - 1)
                    break
                curr_data1 = data1[0].clone().detach(), data1[1].clone().detach()
                curr_data2 = data2[0].clone().detach(), data2[1].clone().detach()
                curr_data3 = data3[0].clone().detach(), data3[1].clone().detach()
                output = model(curr_data1[0].squeeze(1), curr_data2[0].squeeze(1), curr_data3[0].squeeze(1))
                #val_loss = criterion(output[2], output[3], output[0], curr_data[1])
                result = model.loss_function([curr_data1[1].squeeze(1).clone(), curr_data2[1].squeeze(1).clone(), curr_data3[1].squeeze(1).clone()], *output, **{'M_N': 1})
                val_loss = result['loss']
                all_loss += val_loss.item()
            steps += (j + 1)
        curr_loss = all_loss / steps
        if curr_loss < test_loss:
            test_loss = curr_loss
            torch.save(model.state_dict(), 'best_total_time_series1.pt')
            threshold = early_stopping_rounds
        elif curr_loss > test_loss * (1 + tolerance):
            threshold -= 1
            if threshold < 0:
                break
        if i % 2 == 0:
            print(f'{i}-th iteration TEST LOSS :  {curr_loss}')
