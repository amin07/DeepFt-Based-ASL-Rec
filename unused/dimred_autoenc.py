import numpy as np
import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, ConcatDataset
from torch import nn
from data_loader import ASLDataset

class AutoEnc(nn.Module):
  def __init__(self, input_dim, hidden1_len, embd_len):
    super(AutoEnc, self).__init__()
    self.encoder_model = nn.Sequential(nn.Linear(input_len, hidden1_len), nn.ReLU(), nn.Linear(hidden1_len, embd_len))
    self.decoder_model = nn.Sequential(nn.Linear(embd_len, hidden1_len), nn.ReLU(), nn.Linear(hidden1_len, input_len))
    
  def forward(self, x):
    coded = self.encoder_model(x)
    decoded = self.decoder_model(coded)
    return coded, decoded


infer = False
model_name = 'autoenc_300'
num_epochs = 300
batch_size = 500
hidden_size = 2048
num_layers = 1
train_subs = ['paneer','kaleab','ding', 'eddie', 'jensen']
test_subs = ['alamin', 'professor', 'juan', 'aiswarya', 'qian']
learning_rate = 1e-4
sample_rate = 15
embd_len = 256
hidden1_len = 512
input_len = 1024
input_loc = 'embeddings/'
scratch_loc = '/scratch/ahosain/slurm_outs/autoenc_models/'
output_loc = scratch_loc+'reduced_embeddings/'
model_save_loc = '/scratch/ahosain/slurm_outs/autoenc_models/'
model_save_loc += str(input_len)+'_'+str(hidden1_len)+'_'+str(embd_len)
model_path = model_save_loc
if not os.path.isdir(model_save_loc):
  os.mkdir(model_save_loc)


'''
# dataset with small amount of data
embd_files = os.listdir(input_loc)
test_files = [f for f in embd_files if f.split('_')[1] in test_subs]
train_files = [f for f in embd_files if f.split('_')[1] in train_subs]
print ('test classes', len(list(set([f.split('_')[0] for f in test_files]))))
print ('train classes', len(list(set([f.split('_')[0] for f in train_files]))))
class_list = sorted(list(set([f.split('_')[0] for f in test_files])))
num_class = len(class_list)
label2temp = dict([(c[1], c[0]) for c in enumerate(class_list)])
temp2label = dict([c[0], c[1]] for c in enumerate(class_list))
train_x = torch.tensor([np.load(os.path.join(input_loc, f)) for f in train_files])
train_y = torch.tensor([label2temp[f.split('_')[0]] for f in train_files])
train_dataset = TensorDataset(train_x, train_y)
test_x = torch.tensor([np.load(os.path.join(input_loc, f)) for f in test_files])
test_y = torch.tensor([label2temp[f.split('_')[0]] for f in test_files])
test_dataset = TensorDataset(test_x, test_y)
dataset = ConcatDataset((train_dataset, test_dataset))
'''
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if infer==True:
  batch_size = 1
  model = torch.load(os.path.join(model_path, model_name))
  model.eval()
else:
  model = AutoEnc(input_len, hidden1_len, embd_len)

model.to(device)

for p in list(model.parameters()):
  print (p.size())
sys.exit()


loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(list(model.parameters()), lr=learning_rate)

dataset = ASLDataset(input_loc, train_subs+test_subs)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

if infer==True:
  losses = 0.
  out_dir = os.path.join(output_loc, model_name)
  if not os.path.isdir(out_dir):
    os.mkdir(out_dir)
  for i_batch, batch in enumerate(dataloader):
    x_dat = batch['rgb_dat'].reshape((-1, input_len))
    x_dat = x_dat.to(device)
    file_name = batch['file_name'][0]
    effective_batch = x_dat.size(0)
    coded, decoded = model(x_dat)
    np.save(os.path.join(out_dir, file_name), coded.cpu().data.numpy())
    loss = loss_fn(decoded, x_dat)
    losses += loss.item()*effective_batch
  print ('inference loss', losses/len(dataset))
  sys.exit()

for ep in range(num_epochs+1):
  losses = 0.
  for i_batch, batch in enumerate(dataloader):
    #x_dat_left, x_dat_right = batch['rgb_dat'].split(sample_rate, dim=1)
    #x_dat = torch.cat((x_dat_right.float(), x_dat_left.float()), dim=1).squeeze()
    x_dat = batch['rgb_dat'].reshape((-1, input_len))
    x_dat = x_dat.to(device)
    effective_batch = x_dat.size(0)
    coded, decoded = model(x_dat)
    loss = loss_fn(decoded, x_dat)
    losses += loss.item()*effective_batch
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  if ep and ep%20==0:
    torch.save(model, os.path.join(model_save_loc, 'autoenc_'+str(ep)))
  print ('ep', ep+1, 'loss: ', losses/len(dataset))
