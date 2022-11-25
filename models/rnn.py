from torch import nn
import torch
from torch.nn import functional as F
import numpy as np
import pandas as pd
torch.autograd.set_detect_anomaly(True)

class MV_LSTM(nn.Module):
    def __init__(self, n_features, seq_length, n_hidden=20, n_layers=2):
        super().__init__()
        self.n_features = n_features
        self.seq_len = seq_length
        self.n_hidden = n_hidden # number of hidden states
        self.n_layers = n_layers # number of LSTM layers (stacked)
        self.l_lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=self.n_hidden,
            num_layers=self.n_layers,
            batch_first=True
            )
        # according to pytorch docs LSTM output is 
        # (batch_size, seq_len, num_directions * hidden_size)
        # when considering batch_first = True
        self.l_linear = nn.Linear(self.n_hidden * self.seq_len, n_features)
        
    def init_hidden(self, batch_size):
        # even with batch_first = True this remains same as docs
        hidden_state = torch.zeros(self.n_layers, batch_size, self.n_hidden)
        cell_state = torch.zeros(self.n_layers, batch_size, self.n_hidden)
        self.hidden = (hidden_state.detach(), cell_state.detach())
    
    def forward(self, x):        
        batch_size, seq_len, _ = x.size()
        lstm_out, self.hidden = self.l_lstm(x, self.hidden)
        self.hidden = (self.hidden[0].detach(), self.hidden[1].detach())
        # lstm_out(with batch_first = True) is 
        # (batch_size, seq_len, num_directions * hidden_size)
        # for following linear layer we want to keep batch_size dimension and merge rest       
        # .contiguous() -> solves tensor compatibility error
        x = lstm_out.contiguous().view(batch_size, -1)
        return self.l_linear(x)


class CMV_LSTM(nn.Module):
    def __init__(self, n_features1, n_features2, n_features3, seq_length, n_hidden=20, n_layers=2):
        super().__init__()
        self.n_features1 = n_features1
        self.n_features2 = n_features2
        self.n_features3 = n_features3
        self.seq_len = seq_length
        self.n_hidden = n_hidden # number of hidden states
        self.n_layers = n_layers # number of LSTM layers (stacked)
        self.l_lstm1 = nn.LSTM(
            input_size=n_features1 + n_features2 + n_features3,
            hidden_size=self.n_hidden,
            num_layers=self.n_layers,
            batch_first=True
            )
        self.l_lstm2 = nn.LSTM(
            input_size=n_features1 + n_features2 + n_features3,
            hidden_size=self.n_hidden,
            num_layers=self.n_layers,
            batch_first=True
            )
        self.l_lstm3 = nn.LSTM(
            input_size=n_features3,
            hidden_size=self.n_hidden,
            num_layers=self.n_layers,
            batch_first=True
            )
        # according to pytorch docs LSTM output is 
        # (batch_size, seq_len, num_directions * hidden_size)
        # when considering batch_first = True
        self.l_linear1 = nn.Linear(self.n_hidden * self.seq_len, n_features1)
        self.l_linear2 = nn.Linear(self.n_hidden * self.seq_len, n_features2)
        self.l_linear3 = nn.Linear(self.n_hidden * self.seq_len, n_features3)
        
    def init_hidden(self, batch_size):
        # even with batch_first = True this remains same as docs
        hidden_state = torch.zeros(self.n_layers, batch_size, self.n_hidden)
        cell_state = torch.zeros(self.n_layers, batch_size, self.n_hidden)
        self.hidden1 = (hidden_state.detach(), cell_state.detach())
        hidden_state = torch.zeros(self.n_layers, batch_size, self.n_hidden)
        cell_state = torch.zeros(self.n_layers, batch_size, self.n_hidden)
        self.hidden2 = (hidden_state.detach(), cell_state.detach())
        hidden_state = torch.zeros(self.n_layers, batch_size, self.n_hidden)
        cell_state = torch.zeros(self.n_layers, batch_size, self.n_hidden)
        self.hidden3 = (hidden_state.detach(), cell_state.detach())
    
    def forward(self, econs, trends, weathers):        
        batch_size1, seq_len1, _ = econs.size()
        batch_size2, seq_len2, _ = trends.size()
        batch_size3, seq_len3, _ = weathers.size()
        input1 = torch.cat([econs, trends, weathers], dim=2)
        # input2 = torch.cat([trends, weathers], dim=1)
        lstm_out1, self.hidden1 = self.l_lstm1(input1, self.hidden1)
        self.hidden1 = (self.hidden1[0].detach(), self.hidden1[1].detach())
        lstm_out2, self.hidden2 = self.l_lstm2(input1, self.hidden2)
        self.hidden2 = (self.hidden2[0].detach(), self.hidden2[1].detach())
        lstm_out3, self.hidden3 = self.l_lstm3(weathers, self.hidden3)
        self.hidden3 = (self.hidden3[0].detach(), self.hidden3[1].detach())
        # lstm_out(with batch_first = True) is 
        # (batch_size, seq_len, num_directions * hidden_size)
        # for following linear layer we want to keep batch_size dimension and merge rest       
        # .contiguous() -> solves tensor compatibility error
        new_econs = lstm_out1.contiguous().view(batch_size1, -1)
        new_trends = lstm_out2.contiguous().view(batch_size2, -1)
        new_weathers = lstm_out3.contiguous().view(batch_size3, -1)
        return self.l_linear1(new_econs), self.l_linear2(new_trends), self.l_linear3(new_weathers)
