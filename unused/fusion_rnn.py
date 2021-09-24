import numpy as np
import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from data_loader import ASLDataset, PreprocessSample
from network_parameters import *


fusion_type = args.fusion_type
test_model = args.test_model
save_loc = args.save_dir
save_model = args.save_model
num_epochs = args.num_epochs
batch_size = args.batch_size
num_layers = args.num_layers
num_layers_sk = args.num_layers_sk
learning_rate = args.learning_rate
sample_rate = args.sample_rate
sample_rate_sk = args.sample_rate_sk
#input_len = args.input_len
input_loc = args.data_dir
hidden_size = args.state_size
hidden_size_sk = args.state_size_sk
drop_out = args.drop_out


def print_args():
  for arg in vars(args):
    print (arg, getattr(args, arg))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print ('device', device)


class FusionLSTM(torch.nn.Module):
  def __init__(self, input_len, hidden_size, num_layers, num_class,  drop_out=0.5, fusion_type=None):
    super(FusionLSTM, self).__init__()
    self.fusion_type = fusion_type
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.lstm_model = torch.nn.LSTM(input_len[0], hidden_size[0], num_layers[0], batch_first=True, dropout=drop_out)
    self.lstm_model_sk = torch.nn.LSTM(input_len[1], hidden_size[1], num_layers[1], batch_first=True, dropout=drop_out)
    self.smax_layer = torch.nn.Linear(hidden_size[0]+hidden_size[1], num_class)
  def forward(self, x):
    effective_batch = x[0].size(0)
    lstm_out, states = self.lstm_model(x[0])
    _ , cstate = states
    cstate_sep = cstate.view((self.num_layers[0], 1, effective_batch, self.hidden_size[0]))[-1].squeeze()
    
    lstm_out, states_sk = self.lstm_model_sk(x[1])
    _ , cstate_sk = states_sk
    cstate_sep_sk = cstate_sk.view((self.num_layers[1], 1, effective_batch, self.hidden_size[1]))[-1].squeeze()
    
    cstate_sep = torch.cat((cstate_sep, cstate_sep_sk), dim=-1)
    y_logits = self.smax_layer(cstate_sep)
    return y_logits
  
input_path = os.path.join(input_loc)
train_dataset = ASLDataset(input_path, train_subs, transform=PreprocessSample(sample_rate, sample_rate_sk))
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6)
test_dataset = ASLDataset(input_path, test_subs, transform=PreprocessSample(sample_rate, sample_rate_sk))
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=6)

data_sample = test_dataset[0]
sk_input_len = data_sample['sk_frames'].shape[-1]
embd_input_len = data_sample['rgb_dat'].shape[-1]

label2temp, temp2label = test_dataset.get_label_dicts()
num_class = len(label2temp)

if test_model=='': 
  model = FusionLSTM((embd_input_len, sk_input_len), (hidden_size, hidden_size_sk), (num_layers, num_layers_sk), num_class, drop_out, fusion_type=fusion_type)
else:
  model = torch.load(os.path.join(save_loc, test_model))

model.to(device)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(list(model.parameters()), lr=learning_rate)

model_name = '{}_{}_I{}_{}_H{}_{}_L{}_{}'.format(args.fusion_type, args.test_subj, embd_input_len,sk_input_len, hidden_size, hidden_size_sk, num_layers, num_layers_sk)
run_mode = [' ##TRAIN##', ' @@TEST@@ ']
best_test_acc = 0.
best_id = 0
if test_model!='':
  num_epochs = 1
for ep in range(num_epochs):
  for r, dataloader in enumerate([train_dataloader, test_dataloader]):
    if r==0 and test_model!='':
      continue
    ep_loss, ep_acc = [], []
    model.train()      # default train mode
    if r==1:      # evaluation mode for dropout
      model.eval()
    for i_batch, batch in enumerate(dataloader):
      x_dat = batch['rgb_dat'].float()
      y_dat = batch['labels']
      x_dat_sk = batch['sk_frames'].float()
      
      x_dat = x_dat.to(device)
      x_dat_sk = x_dat_sk.to(device)
      y_dat = y_dat.to(device)
      effective_batch = x_dat.size()[0]
      
      y_logits = model((x_dat, x_dat_sk))
      loss = loss_fn(y_logits, y_dat)
      ep_loss.append(loss.item()*effective_batch)
      preds = torch.max(y_logits, dim=-1)[1]
      acc = ((y_dat==preds).float().mean())
      if i_batch%10==0:
        print ('** {} ** batch loss '.format(run_mode[r]), i_batch, loss.item(), "accuracy ", acc.item())
      ep_acc.append(acc.item()*effective_batch)
      data_len = len(test_dataset)
      if r==0:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        data_len = len(train_dataset)
    print ('** {} ** EP: '.format(run_mode[r]), ep, sum(ep_loss)/data_len, sum(ep_acc)/data_len)
    if test_model!='':
      break
    if r:
      if best_test_acc < sum(ep_acc)/data_len:
        best_test_acc = sum(ep_acc)/data_len
        best_id = ep
        if save_model:
          torch.save(model, os.path.join(save_loc, model_name))
      print ('==>> best so far', best_test_acc, best_id, '<==')
      print_args()
print ('best test acc and ep id (during training)', best_test_acc, best_id) 
