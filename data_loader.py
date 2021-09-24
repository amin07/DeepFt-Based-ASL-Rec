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
class OpenposeASLDataset(Dataset):

  def __init__(self, root_dir, subs, transform=None):
    """
    Args:
        root_dir (string): Directory of the files.
        subs (list): subjects to be included in this dataset (test/train)
        transform (callable, optional): Optional transform to be applied
            on a sample.
    """
    all_files = os.listdir(root_dir)
    files = []
    for f in all_files:
      files += [os.path.join(root_dir, f, fn) for fn in os.listdir(os.path.join(root_dir, f))]
    all_files = files
    relevant_files = [f for f in all_files if os.path.basename(f).split('_')[1] in subs and 'embd.npy' in f]   
    class_list = sorted(list(set([os.path.basename(f).split('_')[0] for f in all_files])))   
    self.label2temp = dict([(c[1], c[0]) for c in enumerate(class_list)])
    self.temp2label = dict([(c[0], c[1]) for c in enumerate(class_list)])
    self.file_list = [os.path.join(f) for f in relevant_files]
    self.root_dir = root_dir
    self.transform = transform
    
    print ('considering subjects', subs, 'total files', len(self.file_list))
  def __len__(self):
    return len(self.file_list)

  def __getitem__(self, idx):
    sample = {"rgb_dat":np.load(self.file_list[idx])}
    #print (sample["rgb_dat"].shape)
    sk_file = self.file_list[idx][:-13]+'_bodyPose.npy'
    leftHand_file = self.file_list[idx][:-13]+'_leftHandPose.npy'
    rightHand_file = self.file_list[idx][:-13]+'_rightHandPose.npy'
    sample["sk_frames"] = np.load(sk_file)
    sample["left_hand"] = np.load(leftHand_file)
    sample["right_hand"] = np.load(rightHand_file)
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


class PreprocessOpenEmbd(object):
  def __init__(self, sample_rate, sample_rate_sk=None):
    self.sample_rate_sk = sample_rate_sk
    if self.sample_rate_sk==None:
      self.sample_rate_sk = sample_rate
    self.sample_rate = sample_rate
    #self.joint_idx = [5, 7, 9, 6, 8, 10]
    self.joint_idx = [2, 3, 4, 5, 6, 7]
  def __call__(self, sample):
    sampling_f = lambda m, n: [i*n//m + n//(2*m) for i in range(m)]
    sampled_index = sampling_f(self.sample_rate, sample['frame_count'])
    sampled_index_sk = sampling_f(self.sample_rate_sk, sample['frame_count'])

    sample['sk_frames'] = sample['sk_frames'][sampled_index_sk]
    sample['left_hand'] = sample['left_hand'][sampled_index_sk]
    sample['right_hand'] = sample['right_hand'][sampled_index_sk]

    left_hand_mean = np.mean(sample['left_hand'], axis=1)[:, :2]
    right_hand_mean = np.mean(sample['right_hand'], axis=1)[:, :2]

    imp_joints = sample['sk_frames'][:,self.joint_idx, :2]    # last index is confidence

    #imp_joints[:,2,:] = right_hand_mean
    #imp_joints[:,5,:] = left_hand_mean          # replacing wrist value with mean from hand

    imp_joints = np.concatenate((imp_joints, sample['left_hand'][:,:, :2], sample['right_hand'][:, :, :2]), axis = 1)
    origin = sample['sk_frames'][:,8:9,:2]          # 8 is the mid hip in body format
    sample['sk_frames'] = (imp_joints - origin)*np.array([1., -1.]) # converting y axis because left top is the 0, 0
    #sample['sk_frames'] /= np.array([1080., 1920])      # normalized
    sample['sk_frames'] *= (2.54/9600)       # converting to meter distance

    ### spatial augmentation ###
    left_w_sub = sample['sk_frames'] - sample['sk_frames'][:,2:3, :]
    right_w_sub = sample['sk_frames'] - sample['sk_frames'][:,5:6, :]
    sample['sk_frames'] = sample['sk_frames'].reshape((self.sample_rate_sk, -1))
    left_w_sub = left_w_sub.reshape((self.sample_rate_sk, -1))
    right_w_sub = right_w_sub.reshape((self.sample_rate_sk, -1))
    sample['sk_frames'] = np.concatenate((sample['sk_frames'], left_w_sub, right_w_sub), axis=-1)
    ### spatial augmentation ###
    sample['rgb_dat'] = sample['rgb_dat'][sampled_index]
    return sample
