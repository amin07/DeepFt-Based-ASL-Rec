'''
Created on Oct 21, 2018
LSTM model for sign language recognition
where each joint data fed into one sperate lstm
final embeddings are concatenated to produce classification
@author: Amin
'''
import cv2
import os
import sys
import glob
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt 
import tensorflow as tf
from collections import defaultdict
from sklearn.cluster import KMeans
from image_embedding import get_embeddings
#import graphviz as gv
import argparse
parser = argparse.ArgumentParser()
considered_joints = [4, 5, 6, 8, 9, 10]
considered_joint_ids = [0, 1,2, 3, 4, 5]

# network params
dropout_keep = 0.5
data_loc = '/home/gmuadmin/workspace/deephand/data_224/'
data_loc = '/home/ahosain/asl_new/fusion_lstm/data'
embedding_out_loc = '/scratch/ahosain/slurm_outs/rnn_embd_models/embd_sk_data'
save_folder = 'saves/'
subjects = ['alamin','kaleab','paneer','ding', 'professor','eddie', 'jensen', 'aiswarya', 'juan', 'fatme', 'sofia','qian']
sign_classes = ['ac', 'alarm', 'doorbell', 'snow', 'camera']
sign_classes = ['bedroom', 'calendar', 'cancel', 'day', 'dim','direction','door', 'doorbell','email','event', 'food']
sign_classes = ['game', 'goodmorning', 'heat', 'house', 'kitchen', 'lightbulb', 'list', 'lock', 'message', 'movies']
sign_classes = [ 'night', 'order', 'phone', 'picture', 'place','play','quote','rain','raise','restaurant']
sign_classes = ['room', 'schedule', 'shopping','snooze', 'snow', 'stop', 'sunny', 'temperature', 'time', 'traffic']
sign_classes = ['turndown', 'turnon', 'wakeup','weather', 'weekend', 'work']


parser.add_argument("--class_name", "-cn", help="class name we are considering, visual pps.",type=str)
parser.add_argument("--run_mode", "-rm", help="mode of running script train, test, retrain etc.",type=str)
parser.add_argument("--test_subj","-ts",  help="name or id of the test subject.",type=str)
parser.add_argument("--sample_rate","-sr",  help="sequence len of input.",type=int)
parser.add_argument("--state_size","-ss",  help="state size of the LSTM cells.",type=int)
parser.add_argument("--layer_no","-ln",  help="numbers of layers in LSTM.",type=int, default=1)
parser.add_argument("--run_note","-rn",  help="notes to be displayed while model running.",type=str)
parser.add_argument("--reg_beta","-rb",  help="regularization beta parameter.",type=float, default=0.008)
parser.add_argument("--clip_value","-cv",  help="gradient clipping value.",type=float, default=0.0)
parser.add_argument("--learning_rate","-lr",  help="learning rate of the network.",type=float,default=0.00001)
parser.add_argument("--num_epochs","-ne",  help="training epochs count.",type=int, default=500)
parser.add_argument("--model_name","-mn",  help="model name to test.",type=str)
parser.add_argument("--model_suffix","-ms",  help="model suffix to recognize.",type=str, default="")
parser.add_argument("--good_classes", action='store_true')
parser.add_argument("--motion_sample", action='store_true')
args = parser.parse_args()

reg_beta = args.reg_beta
clip_value = args.clip_value
learning_rate = args.learning_rate


def details_print():
  #print (args.state_size, args.layer_no, args.state_size, args.reg_beta, learning_rate, args.run_note)
  for arg in vars(args):
    print (arg, getattr(args, arg))
reg_beta = args.reg_beta
clip_value = args.clip_value
    

def softmax(x, ax=1):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=ax, keepdims=True)

def calc_velocity(sk_fname, delta=2, srate=5):
  dat_ar = np.load(sk_fname)
  dat_ar = np.split(dat_ar, dat_ar.shape[0], axis=0)
  vel_ar = [100.]*len(dat_ar)
  for i, jd in enumerate(dat_ar[:-2*delta]):
    ci = i+delta
    speed = np.linalg.norm(dat_ar[ci+delta][0][5,:] - dat_ar[ci-delta][0][5,:])
    vel_ar[ci] = speed/(2*delta)
  avg_vel = sum(vel_ar[delta:-delta])/len(vel_ar[delta:-delta])
  sidx = [index for index, value in sorted(enumerate(vel_ar), reverse=True, key=lambda x: x[1])]
  return sorted(sidx[-srate:])

def prep_data_sub(subj, s_rate, class_n='snow'):
  sampling_f = lambda m, n: [i*n//m + n//(2*m) for i in range(m)]
  sk_fnames = [os.path.join(data_loc,'{}/'.format(subj), f) for f in os.listdir(os.path.join(data_loc, '{}'.format(subj))) if f.split('.')[-1]=='npy']
  #sk_fnames = [f for f in sk_fnames if os.path.basename(f).split('_')[0] in sign_classes]
  print (subj, "total sk files", len(sk_fnames))
  list_x = []
  list_y = []
  file_names = []
  for i, f in enumerate(sk_fnames[:]):   
    vid_fname = sk_fnames[i][:-12]+'rgb.avi'
    if not os.path.exists(vid_fname) or not os.path.exists(f):
      continue
    #vel_ids = calc_velocity(f, srate=s_rate)
    vid = cv2.VideoCapture(vid_fname)
    frame_ar = []
    while True:
      ret , frame = vid.read()
      if ret==False:
        break
      frame_ar.append(frame)
    frame_ar = np.array(frame_ar)
    #sample_ids = sampling_f(s_rate, len(frame_ar))
    #if args.motion_sample:
    #  sample_ids = vel_ids
    #try:
    #  sampled_frames = frame_ar[sample_ids,:,:,:]
    #except:
    #  print (frame_ar.shape, f)
    #sampled_frames = np.squeeze(np.concatenate(np.split(sampled_frames, s_rate, axis=0), axis=-3))
    sampled_frames = frame_ar.reshape((-1, 200, 3))       # no sampling or velocity calc
    list_x.append(sampled_frames)
    list_y.append(f.split('/')[-1].split('_')[0])
    file_names.append(os.path.basename(f).split('.')[0])
  get_embeddings(list_x, file_names, 1, s_rate, embedding_out_loc)
  return list_x, list_y  

'''
generated embedding will be saved designated output folder
'''
def make_data(s_rate = 10, class_n = 'weather'):
  for sub in subjects:
    prep_data_sub(sub, s_rate)  

def main():
  make_data(args.sample_rate)

if __name__ == '__main__':
  main()
