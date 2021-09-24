import numpy as np
import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from data_loader import ASLDataset
from network_parameters import *


test_model = args.test_model
save_loc = args.save_dir
save_model = args.save_model
num_epochs = args.num_epochs
batch_size = args.batch_size
num_layers = args.num_layers
learning_rate = args.learning_rate
sample_rate = args.sample_rate
input_len = args.input_len
input_loc = args.data_dir
hidden_size = args.state_size
drop_out = args.drop_out


def print_args():
  for arg in vars(args):
    print (arg, getattr(args, arg))
#sys.exit()
'''
embd_len = 1024*2
scratch_loc = '/scratch/ahosain/slurm_outs/autoenc_models/'
embd_loc = 'embeddings/'
input_loc = os.path.join(scratch_loc, 'reduced_embeddings/')
model_name = 'autoenc_300'
input_len = 256*2

input_len = embd_len
input_loc = embd_loc
model_name = ''
'''
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print ('device', device)
'''
embd_files = os.listdir(embd_loc)
test_files = [f for f in embd_files if f.split('_')[1] in test_subs]
train_files = [f for f in embd_files if f.split('_')[1] in train_subs]
print ('test classes', len(list(set([f.split('_')[0] for f in test_files]))))
print ('train classes', len(list(set([f.split('_')[0] for f in train_files]))))
class_list = sorted(list(set([f.split('_')[0] for f in test_files])))
num_class = len(class_list)
label2temp = dict([(c[1], c[0]) for c in enumerate(class_list)])
temp2label = dict([c[0], c[1]] for c in enumerate(class_list))
lstm_model = torch.nn.LSTM(input_len, hidden_size, num_layers, batch_first=True)
smax_layer = torch.nn.Linear(hidden_size, num_class)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(list(lstm_model.parameters())+list(smax_layer.parameters()), lr=learning_rate)
train_x = torch.tensor([np.load(os.path.join(embd_loc, f)) for f in train_files])
train_y = torch.tensor([label2temp[f.split('_')[0]] for f in train_files])
train_dataset = TensorDataset(train_x, train_y)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_x = torch.tensor([np.load(os.path.join(embd_loc, f)) for f in test_files])
test_y = torch.tensor([label2temp[f.split('_')[0]] for f in test_files])
test_dataset = TensorDataset(test_x, test_y)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
'''
class EmbdLSTM(torch.nn.Module):
  def __init__(self, input_len, hidden_size, num_layers, num_class,  drop_out=0.5):
    super(EmbdLSTM, self).__init__()
    self.num_layers = num_layers
    self.hidden_size = hidden_size
    self.lstm_model = torch.nn.LSTM(input_len, hidden_size, num_layers, batch_first=True, dropout=drop_out)
    self.smax_layer = torch.nn.Linear(hidden_size, num_class)
  def forward(self, x):
    effective_batch = x.size(0)
    lstm_out, states = self.lstm_model(x)
    _ , cstate = states
    cstate_sep = cstate.view((self.num_layers, 1, effective_batch, self.hidden_size))[-1].squeeze()
    y_logits = self.smax_layer(cstate_sep)
    return y_logits
  
input_path = os.path.join(input_loc)
train_dataset = ASLDataset(input_path, train_subs)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6)
test_dataset = ASLDataset(input_path, test_subs)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=6)

label2temp, temp2label = test_dataset.get_label_dicts()
num_class = len(label2temp)
'''
lstm_model = torch.nn.LSTM(input_len, hidden_size, num_layers, batch_first=True, dropout=drop_out)
lstm_model.to(device)
smax_layer = torch.nn.Linear(hidden_size, num_class)
smax_layer.to(device)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(list(lstm_model.parameters())+list(smax_layer.parameters()), lr=learning_rate)
'''

if test_model=='': 
  model = EmbdLSTM(input_len, hidden_size, num_layers, num_class, drop_out)
else:
  model = torch.load(os.path.join(save_loc, test_model))

model.to(device)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(list(model.parameters()), lr=learning_rate)

model_name = '{}_I{}_H{}_L{}'.format(args.test_subj, input_len, hidden_size, num_layers)
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
      #x_dat_right, x_dat_left = batch['rgb_dat'].split(sample_rate, dim=1)
      #x_dat = torch.cat((x_dat_right.float(), x_dat_left.float()), dim=-1)
      x_dat = batch['rgb_dat']
      y_dat = batch['labels']
      x_dat = x_dat.to(device)
      y_dat = y_dat.to(device)
      effective_batch = x_dat.size()[0]
      #lstm_out, states = lstm_model(x_dat)
      #_ , cstate = states
      #cstate_sep = cstate.view((num_layers, 1, effective_batch, hidden_size))[-1].squeeze()
      #y_logits = smax_layer(cstate_sep)
      y_logits = model(x_dat)
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
