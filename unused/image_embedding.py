#!/usr/bin/env python

from __future__ import print_function

import os
import cv2
import numpy as np
import tensorflow as tf
#import utils
import time
import sys
import platform

# Define Code and Data Path
# code_path: where the classify.py resides
# data_path: where the images folder of eval set resides

# this three line for desktop pc
#code_path = '/home/gmuadmin/workspace/deephand/TF-DeepHand/'
#data_path = '/home/gmuadmin/workspace/deephand/TF-DeepHand/data/ph2014-dev-set-handshape-annotations/'
#model_path = code_path + 'deephand/deephand_model.npy';

# this lines for cluster
#code_path = '/'
model_path = os.path.join('deephand/deephand_model.npy')

# For Windows
if platform.system() == 'Windows':
    sys.path.append(code_path+'deephand')
    
from deephand.deephand import DeepHand
'''
# Define Batch Size
batch_size = 8;

# input_list_file: a file with relative paths of images that you want to evaluate
input_list_file = code_path + 'input/3359-ph2014-MS-handshape-index.txt'
# mean_path: mean image path
mean_path = code_path + 'input/onemilhands_mean.npy';
# model_path: pretrained weights path
model_path = code_path + 'deephand/deephand_model.npy';

# Define Network
input_node = tf.placeholder(tf.float32, shape=(None, 224, 224, 3));
net = DeepHand({'data': input_node})
embedding_vector = tf.get_default_graph().get_tensor_by_name("pool5_7x7_s1:0")

# Load Mean Image
mean = np.load(mean_path);
mean = np.transpose(mean, (1, 2, 0));

# Read Image List and Labels
image_paths, labels = utils.read_eval_image_list(input_list_file)

# Get Number of Iterations
num_samples = len(labels);
num_iter = np.int(np.ceil(1.0*num_samples/batch_size));

# Create the storage for image_scores
image_scores = np.zeros((num_iter*batch_size,61))
'''

input_node = tf.placeholder(tf.float32, shape=(None, 224, 224, 3));
net = DeepHand({'data': input_node})
embedding_vector = tf.get_default_graph().get_tensor_by_name("pool5_7x7_s1:0")
def get_embeddings(image_list, file_names, batch_size=1, sr=1, out_loc=''):
  # Define Network
  num_samples = len(image_list)
  num_iter = np.int(np.ceil(1.0*num_samples/batch_size));
  # Set TF-Session Config Parameters
  config=tf.ConfigProto(allow_soft_placement=True)
  config.gpu_options.allow_growth=True
  # Create a session and get posteriors from each frame
  with tf.Session(config=config) as sesh:
    # Set Device
    with tf.device("/gpu:0"):
      # Load Network Weights
      net.load(model_path, sesh)
      # Start Timer
      begin = time.time();
      sample_iter = 0;
      for i in range(0, num_samples):
        batched_input = image_list[i]
        left_hand_input, right_hand_input = np.split(batched_input, 2, axis=1)   # seperating left/right
        sr = batched_input.shape[0]     # sr here is all frames, just keeping all, 100 is the height
        outfile = os.path.join(out_loc, file_names[i])
        if sr == 0:
          print ('FRAME COUNT 0, continue..')
          continue
        if os.path.isfile(outfile+'.npy'):
          print ('Exist, continue')
          continue
        total_inputs = np.split(left_hand_input, sr) + np.split(right_hand_input, sr)   
        batched_input = [np.squeeze(ti) for ti in total_inputs]
        batched_input = np.array([cv2.resize(f, (224, 224)) for f in batched_input])   # one sample of sr 5 gives 10 patches
        #cv2.imshow('frame', image_list[i])
        #cv2.waitKey(-1)
        sess_ret = sesh.run([net.get_output(), embedding_vector], feed_dict={input_node: batched_input})
        embeddings = sess_ret[1].squeeze()
        print ('processed ', i, len(sess_ret[0].shape), sess_ret[1].shape, sess_ret[0].shape)
        curr = time.time()-begin;
        print("Evaluated {}/{} iterations in {:.2f} seconds - {:.2f} seconds/iteration".format(i+1, num_iter, curr, curr/(i+1)))
        np.save(outfile, embeddings)
      elapsed = time.time()-begin;
      print("Total Evaluation Time: {:.2f} seconds".format(elapsed))

  #sys.exit()

  # Only get the valid scores
  #image_scores = image_scores[0:num_samples,:];

  # Get predictions
  #predictions = np.argmax(image_scores, axis=1);

  # Get Accuracy
  #accuracy = 100.0 * np.sum(predictions == np.int64(np.asarray(labels))) / len(predictions);

  # Print out
  #print("Accruracy on Test Set: {:.4f}".format(accuracy))
