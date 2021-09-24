import numpy as np
import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset

class AttConcatFusionLSTM(torch.nn.Module):
  def __init__(self, input_len, hidden_size, num_layers, num_class,  drop_out=0.5):
    super(AttConcatFusionLSTM, self).__init__()
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.lstm_model = torch.nn.LSTM(input_len[0], hidden_size[0], num_layers[0], batch_first=True, dropout=drop_out)
    self.lstm_model_sk = torch.nn.LSTM(input_len[1], hidden_size[1], num_layers[1], batch_first=True, dropout=drop_out)
    self.smax_layer = torch.nn.Linear(hidden_size[0]+hidden_size[1], num_class)
    self.score_param = torch.nn.Sequential(torch.nn.Linear(hidden_size[0]+hidden_size[1], 512), torch.nn.Tanh(), torch.nn.Linear(512, 2))
  def forward(self, x):
    effective_batch = x[0].size(0)
    lstm_out, states = self.lstm_model(x[0])
    _ , cstate = states
    cstate_sep = cstate.view((self.num_layers[0], 1, effective_batch, self.hidden_size[0]))[-1][0]

    lstm_out, states_sk = self.lstm_model_sk(x[1])
    _ , cstate_sk = states_sk
    cstate_sep_sk = cstate_sk.view((self.num_layers[1], 1, effective_batch, self.hidden_size[1]))[-1][0]
    
    ### focus layer ### 
    scores = self.score_param(torch.cat((cstate_sep, cstate_sep_sk), dim=-1))
    cstate_sep = torch.cat((scores[:, 0:1]*cstate_sep, scores[:, 1:]*cstate_sep_sk), dim=-1)
    
    y_logits = self.smax_layer(cstate_sep)
    return y_logits, scores


class ConcatFusionLSTM(torch.nn.Module):
  def __init__(self, input_len, hidden_size, num_layers, num_class,  drop_out=0.5):
    super(ConcatFusionLSTM, self).__init__()
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.lstm_model = torch.nn.LSTM(input_len[0], hidden_size[0], num_layers[0], batch_first=True, dropout=drop_out)
    self.lstm_model_sk = torch.nn.LSTM(input_len[1], hidden_size[1], num_layers[1], batch_first=True, dropout=drop_out)
    self.smax_layer = torch.nn.Linear(hidden_size[0]+hidden_size[1], num_class)
  def forward(self, x):
    effective_batch = x[0].size(0)
    lstm_out, states = self.lstm_model(x[0])
    _ , cstate = states
    cstate_sep = cstate.view((self.num_layers[0], 1, effective_batch, self.hidden_size[0]))[-1][0]
    
    lstm_out, states_sk = self.lstm_model_sk(x[1])
    _ , cstate_sk = states_sk
    cstate_sep_sk = cstate_sk.view((self.num_layers[1], 1, effective_batch, self.hidden_size[1]))[-1][0]
    
    cstate_sep = torch.cat((cstate_sep, cstate_sep_sk), dim=-1)
    y_logits = self.smax_layer(cstate_sep)
    return y_logits


class LogitFusionLSTM(torch.nn.Module):
  def __init__(self, input_len, hidden_size, num_layers, num_class,  drop_out=0.5, fusion_type=None):
    super(LogitFusionLSTM, self).__init__()
    self.fusion_type = fusion_type
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.lstm_model = torch.nn.LSTM(input_len[0], hidden_size[0], num_layers[0], batch_first=True, dropout=drop_out)
    self.lstm_model_sk = torch.nn.LSTM(input_len[1], hidden_size[1], num_layers[1], batch_first=True, dropout=drop_out)
    self.smax_layer = torch.nn.Linear(hidden_size[0], num_class)
    self.smax_layer_sk = torch.nn.Linear(hidden_size[1], num_class)

  def forward(self, x):
    effective_batch = x[0].size(0)
    lstm_out, states = self.lstm_model(x[0])
    _ , cstate = states
    cstate_sep = cstate.view((self.num_layers[0], 1, effective_batch, self.hidden_size[0]))[-1][0]
    lstm_out, states_sk = self.lstm_model_sk(x[1])
    _ , cstate_sk = states_sk
    cstate_sep_sk = cstate_sk.view((self.num_layers[1], 1, effective_batch, self.hidden_size[1]))[-1][0]
    y_logits = self.smax_layer(cstate_sep)
    y_logits_sk = self.smax_layer_sk(cstate_sep_sk)
    
    if self.fusion_type=='sum':
      y_logits = (y_logits*args.concat_score) + (1.-args.concat_score)*y_logits_sk
    if self.fusion_type=='mean':
      y_logits = (y_logits + y_logits_sk)/2.
    if self.fusion_type=='only_embd':
      y_logits = y_logits
    if self.fusion_type=='only_sk':
      y_logits = y_logits_sk
    return y_logits
