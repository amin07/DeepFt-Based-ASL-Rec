from __future__ import print_function, division
import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from collections import defaultdict
'''
script file for loading skeletal data for har
'''

class ASLDataset(Dataset):

  """Face Landmarks dataset."""
  def __init__(self, root_dir, subs, transform=None):
    """
    Args:
        root_dir (string): Directory of the files.
        subs (list): subjects to be included in this dataset (test/train)
        transform (callable, optional): Optional transform to be applied
            on a sample.
    """
    all_files = os.listdir(root_dir)
    relevant_files = [f for f in all_files if f.split('_')[1] in subs and 'embedding' in f]   
    #relevant_files = [f for f in all_files if f.split('_')[1] in subs]   # for only embd location
    class_list = sorted(list(set([f.split('_')[0] for f in all_files])))   
    self.label2temp = dict([(c[1], c[0]) for c in enumerate(class_list)])
    self.temp2label = dict([(c[0], c[1]) for c in enumerate(class_list)])
    self.file_list = [os.path.join(root_dir, f) for f in relevant_files]
    self.root_dir = root_dir
    self.transform = transform

  def __len__(self):
    return len(self.file_list)

  def __getitem__(self, idx):
    sample = {"rgb_dat":np.load(self.file_list[idx])}
    sk_file = self.file_list[idx][:-13]+'bodyData.npy'
    sample["sk_frames"] = np.load(sk_file)
    sample['labels'] = self.label2temp[os.path.basename(self.file_list[idx]).split('_')[0]]
    sample['text_labels'] = self.temp2label[sample['labels']]
    sample['file_name'] = os.path.basename(self.file_list[idx])
    sample['frame_count'] = sample['sk_frames'].shape[0]
    sample['rgb_dat'] = np.concatenate(np.split(sample['rgb_dat'], 2), axis=-1) # splitting left/right and cat   
    if self.transform:
        sample = self.transform(sample)
    return sample

  def get_label_dicts(self):
    return dict(self.label2temp), dict(self.temp2label)

class PreprocessSample(object):
  def __init__(self, sample_rate, sample_rate_sk=None):
    self.sample_rate_sk = sample_rate_sk
    if self.sample_rate_sk==None:
      self.sample_rate_sk = sample_rate
    self.sample_rate = sample_rate
  def __call__(self, sample):
    sampling_f = lambda m, n: [i*n//m + n//(2*m) for i in range(m)]
    sampled_index = sampling_f(self.sample_rate, sample['frame_count'])
    sampled_index_sk = sampling_f(self.sample_rate_sk, sample['frame_count'])
    sample['sk_frames'] = sample['sk_frames'][sampled_index_sk].reshape((self.sample_rate_sk, -1))
    sample['rgb_dat'] = sample['rgb_dat'][sampled_index]
    return sample

'''
sub_list = ['kaleab','alamin','paneer','ding']
asl_dataset = ASLDataset('/home/gmuadmin/workspace/deephand/TF-DeepHand/embeddings/', sub_list)
dataloader = DataLoader(asl_dataset, batch_size=10, shuffle=True, num_workers=4)
print (len(asl_dataset))
for i_batch, batch in enumerate(dataloader):
  print (batch['rgb_dat'].shape)
  print (batch['labels'])
  print (batch['text_labels'])
  sys.exit()
'''
