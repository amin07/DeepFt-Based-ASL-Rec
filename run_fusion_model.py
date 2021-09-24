import numpy as np
import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from data_loader import OpenposeASLDataset, PreprocessOpenEmbd
from model_classes import ConcatFusionLSTM, LogitFusionLSTM, AttConcatFusionLSTM
import argparse
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

parser = argparse.ArgumentParser()
parser.add_argument("--save_dir", "-sd", help="directory where models to be saved", type=str, default='saves/')
parser.add_argument("--test_model", "-tm", help="full path to the model to be tested.",type=str, default='')
parser.add_argument("--fusion_type", "-ft", help="type of fusion.",type=str, default='focus_concat')
parser.add_argument("--test_subj","-ts",  help="name or id of the test subject.",type=str, default=None)
parser.add_argument("--data_dir", "-dd", help="location of input embeddings. default to original embd.",type=str, default='deephand_embeddings/')
parser.add_argument("--input_len","-il",  help="len of input dimension. default is embedding size 2*1024",type=int, default=2*1024)
parser.add_argument("--state_size","-ss",  help="state size of lstm. default double of input len.",type=int, default=1024)
parser.add_argument("--state_size_sk","-sssk",  help="state size of lstm. default double of input len. sk",type=int, default=200)
parser.add_argument("--num_layers","-nl",  help="number of layers.",type=int, default=2)
parser.add_argument("--num_layers_sk","-nlsk",  help="number of layers sk lstm.",type=int, default=2)
parser.add_argument("--sample_rate","-sr",  help="frames considered per sample.",type=int, default=15)
parser.add_argument("--sample_rate_sk","-srsk",  help="frames considered per sample sk.",type=int, default=20)
parser.add_argument("--num_epochs","-ne",  help="number of epochs.",type=int, default=50)
parser.add_argument("--batch_size","-bs",  help="size of each batch.",type=int, default=64)
parser.add_argument("--drop_out","-do",  help="drop out keep probability.",type=float, default=0.5)
parser.add_argument("--learning_rate","-le",  help="learning rate.",type=float, default=0.001)
parser.add_argument("--save_model",  help="if save model or not.", action='store_true')

args = parser.parse_args()

if not args.test_subj and not args.test_model:
  print ('Error: no test subject specified in training!')
elif not args.test_subj:
  args.test_subj = args.test_model.split('_')[0]

if args.state_size==None:
  args.state_size = args.input_len*2

all_subs = ['subject{:02d}'.format(i) for i in range(1, 13)]
train_subs = [s for s in all_subs if s!=args.test_subj]
test_subs = [args.test_subj]



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
input_loc = args.data_dir
hidden_size = args.state_size
hidden_size_sk = args.state_size_sk
drop_out = args.drop_out


def print_args():
  for arg in vars(args):
    print (arg, getattr(args, arg))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print ('device', device)

input_path = os.path.join(input_loc)

print (input_path)
train_dataset = OpenposeASLDataset(input_path, train_subs, transform=PreprocessOpenEmbd(sample_rate, sample_rate_sk))
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6)
test_dataset = OpenposeASLDataset(input_path, test_subs, transform=PreprocessOpenEmbd(sample_rate, sample_rate_sk))
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=6)

data_sample = test_dataset[0]
sk_input_len = data_sample['sk_frames'].shape[-1]
embd_input_len = data_sample['rgb_dat'].shape[-1]

label2temp, temp2label = test_dataset.get_label_dicts()
num_class = len(label2temp)

if test_model=='': 
  if fusion_type=='focus_concat':
    model = AttConcatFusionLSTM((embd_input_len, sk_input_len), (hidden_size, hidden_size_sk), (num_layers, num_layers_sk), num_class, drop_out)
  if fusion_type=='concat':
    model = ConcatFusionLSTM((embd_input_len, sk_input_len), (hidden_size, hidden_size_sk), (num_layers, num_layers_sk), num_class, drop_out)
  if fusion_type=='sum':
    model = LogitFusionLSTM((embd_input_len, sk_input_len), (hidden_size, hidden_size_sk), (num_layers, num_layers_sk), num_class, drop_out, fusion_type=fusion_type)
  if fusion_type=='mean':
    model = LogitFusionLSTM((embd_input_len, sk_input_len), (hidden_size, hidden_size_sk), (num_layers, num_layers_sk), num_class, drop_out, fusion_type=fusion_type)
  if fusion_type=='only_sk':
    model = LogitFusionLSTM((embd_input_len, sk_input_len), (hidden_size, hidden_size_sk), (num_layers, num_layers_sk), num_class, drop_out, fusion_type=fusion_type)
  if fusion_type=='only_embd':
    model = LogitFusionLSTM((embd_input_len, sk_input_len), (hidden_size, hidden_size_sk), (num_layers, num_layers_sk), num_class, drop_out, fusion_type=fusion_type)
else:
  model = torch.load(os.path.join(test_model))

model.to(device)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(list(model.parameters()), lr=learning_rate)

model_name = '{}_{}_I{}_{}_H{}_{}_L{}_{}'.format(fusion_type, args.test_subj, embd_input_len,sk_input_len, hidden_size, hidden_size_sk, num_layers, num_layers_sk)
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
      if args.fusion_type=='focus_concat':
        y_logits, scores = model((x_dat, x_dat_sk))
      else:
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
    print ('** {} ** EP: '.format(run_mode[r]), ep, sum(ep_loss)/data_len, sum(ep_acc)/data_len, flush=True)
    if test_model!='':
      break
    if r:
      if best_test_acc < sum(ep_acc)/data_len:
        best_test_acc = sum(ep_acc)/data_len
        best_id = ep
        if save_model:
          torch.save(model, os.path.join(save_loc, model_name))
      print ('==>> best so far', best_test_acc, best_id, '<==', flush=True)
      print_args()
if test_model=='':
  print ('best test acc and ep id (during training)', best_test_acc, best_id) 
