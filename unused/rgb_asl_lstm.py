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
import graphviz as gv
import argparse
parser = argparse.ArgumentParser()
considered_joints = [4, 5, 6, 8, 9, 10]
considered_joint_ids = [0, 1,2, 3, 4, 5]

# network params
dropout_keep = 0.5
data_loc = '/home/gmuadmin/workspace/deephand/data_224/'
data_loc = '/home/gmuadmin/workspace/asl_new/fusion_lstm/data/'
save_folder = 'saves/'

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
parser.add_argument("--num_clusters", "-nc", help="number of clsuters in k means.", type=int, default=10)
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
  sk_fnames = [data_loc+'{}/'.format(subj)+f for f in os.listdir(data_loc+'{}'.format(subj)) if f.split('.')[-1]=='npy']
  #sk_fnames = [f for f in sk_fnames if os.path.basename(f).split('_')[0]==class_n]
  list_x = []
  list_y = []
    
  for i, f in enumerate(sk_fnames[:]):   
    vid_fname = sk_fnames[i][:-12]+'rgb.avi'
    if not os.path.exists(vid_fname) or not os.path.exists(f):
      continue
    vel_ids = calc_velocity(f, srate=s_rate)
    vid = cv2.VideoCapture(vid_fname)
    frame_ar = []
    while True:
      ret , frame = vid.read()
      if ret==False:
        break
      frame_ar.append(frame)
    frame_ar = np.array(frame_ar)
    sample_ids = sampling_f(s_rate, len(frame_ar))
    if args.motion_sample:
      sample_ids = vel_ids
    try:
      sampled_frames = frame_ar[sample_ids,:,:,:]
    except:
      print (frame_ar.shape, f)
    sampled_frames = np.squeeze(np.concatenate(np.split(sampled_frames, s_rate, axis=0), axis=-3))
    list_x.append(sampled_frames)
    list_y.append(f.split('/')[-1].split('_')[0])
  print ('here')
  get_embeddings(list_x, 1, s_rate)
  return list_x, list_y  

def make_data(test_subj, s_rate = 10, class_n = 'weather'):
  '''
  sampling_f = lambda m, n: [i*n//m + n//(2*m) for i in range(m)]
  train_subs = ['alamin', 'ding', 'qian', 'jensen','paneer']
  train_subs = [s for s  in train_subs if s!=test_subj]
  print (train_subs)
  #sk_fnames = ['data/{}/{}_{}_5_bodyData.npy'.format(s, args.class_name,s) for s in os.listdir('data/')]
  sk_fnames = ['data/{}/{}_{}_5_bodyData.npy'.format(args.test_subj, args.class_name,args.test_subj)]
  sample_list = []
  for i, f in enumerate(sk_fnames):
    print (f)
    #test_list_y.append(f.split('/')[-1].split('_')[0])
    sk_ar = np.load(sk_fnames[i])
    vid_fname = sk_fnames[i][:-12]+'rgb.avi'
    vid = cv2.VideoCapture(vid_fname)
    frame_ar = []
    while True:
      ret , frame = vid.read()
      if ret==False:
        break
      frame_ar.append(frame)
    frame_ar = np.array(frame_ar)
    sample_ids = sampling_f(s_rate, len(frame_ar))
    #print (sample_ids)
    sampled_frames = frame_ar[sample_ids,:,:,:]
    sample_list.append(sampled_frames)
    #sampled_frames = np.concatenate(np.split(sampled_frames, s_rate), axis=-3)
    #test_list_x.append(sampled_frames)
  #print (len(test_list_x), test_list_x[0].shape)
  print (sample_list[-1].shape)
  total_ar = np.concatenate(sample_list[-1], axis=-3)
  cv2.imshow('frames',total_ar)
  cv2.waitKey(-1)
  sys.exit()
  '''
  test_list_x, test_list_y = prep_data_sub(test_subj, s_rate, class_n)
  avail_classes = set(test_list_y)
  test_x = np.array(test_list_x)
  class_list = sorted(list(avail_classes))
  print (test_x.shape, len(test_list_y))
  print (avail_classes)
  train_list_x, train_list_y = [], []
  for ts in train_subs:
    trains_x, trains_y = prep_data_sub(ts, s_rate, class_n)
    trains_x = [(t[0],t[1]) for t in zip(trains_x, trains_y) if t[1] in avail_classes]
    trains_y = [t[1] for t in trains_x]
    trains_x = [t[0] for t in trains_x]
    train_list_x += trains_x
    train_list_y += trains_y
  
  train_x = np.array(train_list_x)
  print (train_x.shape, len(train_list_y))
  #train_subs = [test_subj]
  NUM_CLASS = len(class_list)
  label2temp = dict([(c[1], c[0]) for c in enumerate(class_list)])
  temp2label = dict([c[0], c[1]] for c in enumerate(class_list))
  for k in label2temp.keys():
    print (k, label2temp[k])
  
  test_y = np.array([label2temp[k] for k in test_list_y])
  test_y = np.expand_dims(test_y, axis=-1)
  train_y = np.array([label2temp[k] for k in train_list_y])
  train_y = np.expand_dims(train_y, axis=-1)
  return train_x, train_y, test_x, test_y, label2temp, temp2label, class_list

def plot_conf_graph(cm='', clist=[], ts=''):
  print (type(cm))
  print (clist)

  g = gv.Digraph('G', filename='conf_graph_{}.gv'.format(ts))
  r, c = cm.shape
  for i in range(r):
    for j in range(c):
      if i!=j and cm[i,j]>=5:
       g.edge(clist[i], clist[j], label=str(cm[i,j]))
        
  g.view()

def show_image(im, fname=''):
  cv2.imshow(fname, im.astype(np.uint8))
  cv2.waitKey(-1)
def conv_lstm(graph_scope='conv_lstm', input_='', NUM_CLASS=10):
  print ('input shape',input_.get_shape())
  with tf.variable_scope(graph_scope):
    cell = tf.contrib.rnn.ConvLSTMCell(2, [100,100,3], 3, [3,3])
    rnn_outputs, rnn_state = tf.nn.dynamic_rnn(cell, input_, dtype=tf.float32)
    print (type(rnn_outputs), type(rnn_state))
    conv1 = tf.layers.conv2d(rnn_state[0], 10, (3,3))
    print (conv1.get_shape())
    conv1 = tf.layers.conv2d(conv1, 10, (3,3))
    print (conv1.get_shape())
    conv1 = tf.layers.conv2d(conv1, 5, (3,3))
    print (conv1.get_shape())
    conv1 = tf.layers.max_pooling2d(conv1, (3,3), (3,3))
    print (conv1.get_shape())
    conv1 = tf.layers.conv2d(conv1, 3, (3,3))
    print (conv1.get_shape())
    conv1 = tf.layers.max_pooling2d(conv1, (3,3), (3,3))
    print (conv1.get_shape())
    final_state = tf.layers.flatten(conv1)
    print (final_state.get_shape())
    final_logits = tf.layers.dense(final_state, NUM_CLASS)
    print (final_logits.get_shape())
  return final_logits 

def window_cnn(input_='', scope='', reg_beta=0.008):
  reg = tf.contrib.layers.l2_regularizer(reg_beta)
  with tf.variable_scope(scope):
    conv1 = tf.layers.conv2d(input_, 10, (3,3), activation=tf.nn.relu, kernel_regularizer=reg)
    print (conv1.get_shape())
    conv1 = tf.layers.conv2d(conv1, 10, (3,3),activation=tf.nn.relu, kernel_regularizer=reg)
    print (conv1.get_shape())
    conv1 = tf.layers.conv2d(conv1, 5, (3,3), activation=tf.nn.relu, kernel_regularizer=reg)
    print (conv1.get_shape())
    conv1 = tf.layers.max_pooling2d(conv1, (3,3), (3,3))
    print (conv1.get_shape())
    conv1 = tf.layers.conv2d(conv1, 3, (3,3), activation=tf.nn.relu, kernel_regularizer=reg)
    print (conv1.get_shape())
    conv1 = tf.layers.max_pooling2d(conv1, (3,3), (3,3))
    print (conv1.get_shape())
    final_state = tf.layers.flatten(conv1)
    print (final_state.get_shape())
    #final_logits = tf.layers.dense(final_state, NUM_CLASS)
    #print (final_logits.get_shape())
  return final_state

def tensorflow_model(nepochs=500, batch_size=20, hidden_size=20, sample_rate=15, run_mode='', \
        saved_model_name='', test_subj='', num_layers=2):

  train_x, train_y, test_x, test_y, class_list, label2temp, temp2label = make_data(test_subj, s_rate=sample_rate, class_n=args.class_name)
  print (train_x.shape, train_y.shape, test_x.shape, test_y.shape)  
 
  
  dispi = 0
  while dispi<train_x.shape[0]:
    #rand_id = np.random.randint(0, train_x.shape[0])
    rand_id = dispi
    frame_dat = train_x[rand_id,:,:,:]
    print (frame_dat.shape)
    concat_dat = np.squeeze(np.split(frame_dat, sample_rate), axis=0)
    print (concat_dat[0].shape)
    concat_dat = np.concatenate(concat_dat, axis=0)
    print (concat_dat.shape)
    print (train_y[rand_id])
    print ('class label', temp2label[train_y[rand_id,0]])
    #concat_dat = cv2.cvtColor(concat_dat, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame'+str(dispi), concat_dat)
    cv2.waitKey(-1) 
    dispi += 1
  sys.exit()
  
  NUM_CLASS = len(class_list)
  ############ TF GRAPH ######################  
  
  tf.reset_default_graph()
  x_ph = tf.placeholder(tf.float32,(None, sample_rate, 100, 200, 3))
  y_ph = tf.placeholder(tf.uint8,(None, 1))
  keep_prob_ph = tf.placeholder(tf.float32, name='keep_prob_ph')
  inputs = tf.split(x_ph, 2, axis=-2)
    #final_logits = conv_lstm(input_=inputs[1], NUM_CLASS=NUM_CLASS)
  inputs = tf.split(inputs[0], 2, axis=1) + tf.split(inputs[1], 2, axis=1)
  print ('inputs[0]', inputs[0].get_shape())
  inputs = [i[:,0,:,:,:] for i in inputs]
  print ('inputs[0]', inputs[0].get_shape())
  print (len(inputs))
  test_image = inputs[args.num_clusters]
  concat_state = []
  for i, ins in enumerate(inputs):
      concat_state.append(window_cnn(inputs[i], 'cnn_'+str(i)))
  concat_state = tf.concat(concat_state, axis=-1)  
  print (concat_state.get_shape(), "concat state shape")
  final_logits = tf.layers.dense(concat_state, NUM_CLASS)
  #sys.exit()
  with tf.name_scope("loss_comp"):
    data_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=final_logits, labels=tf.one_hot(y_ph, NUM_CLASS)))
    regularizers = 0.
    reg_loss = tf.losses.get_regularization_loss()
    total_loss = tf.reduce_mean(data_loss + reg_beta * regularizers)
    total_loss += reg_loss
    #total_loss = tf.reduce_mean(data_loss)
    
  with tf.name_scope("train_step"):
    optzr = tf.train.AdamOptimizer(learning_rate)
    gs, vs = zip(*optzr.compute_gradients(total_loss))
    #_, global_norm = tf.clip_by_global_norm(gs, clip_value)
    train_step = optzr.apply_gradients(zip(gs, vs))
   
  scaled_probs = tf.nn.softmax(final_logits)
  predictions = tf.reshape(tf.argmax(scaled_probs, 1), [-1, 1])
  true_labels = tf.cast(y_ph, tf.int64)
  correct_preds = tf.equal(predictions, true_labels)
  accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32))
  ############ TF GRAPH ENDS ######################  
  saver = tf.train.Saver(max_to_keep = None)  # maximum all checkpoint would be retained
  checkpoint = '{}/'.format(save_folder)
 
  if run_mode=='test':
    sess = tf.Session()
    saver.restore(sess, checkpoint+saved_model_name)
    test_accuracy = 0.
    test_losses = 0.
    top_k_acc = defaultdict(lambda : 0)
    pred_list, target_list = [], []
    for sample_no in range(test_x.shape[0]):
      batch_x = test_x[sample_no:sample_no+1,:,:,:]
      batch_x_final = batch_x.reshape(1, sample_rate, -1)
      batch_y = test_y[sample_no:sample_no+1]
      test_loss, test_acc, prob_dist = sess.run([total_loss, accuracy, scaled_probs], feed_dict={x_ph:batch_x_final, y_ph:batch_y, keep_prob_ph:1.0})
      test_accuracy += test_acc
      test_losses += test_loss 
      top_5 = np.argsort(prob_dist)[:,-5:]
      for top_acc in range(1, 6):
        top_k_acc[top_acc] += np.any(top_5[:,-top_acc:]==batch_y)
      pred_list.append(temp2label[top_5[0,-1]])
      target_list.append(temp2label[batch_y[0,-1]])
      #print (batch_y, temp2label[batch_y[0][0]], top_5[:, -1])
    print ('tok 5 accuracy')
    for k in top_k_acc.keys():
      print (k, top_k_acc[k]/test_x.shape[0])
    cnf_matrix = confusion_matrix(target_list, pred_list, labels=class_list)
    plot_conf_graph(cnf_matrix, class_list, ts=test_subj)
    np.set_printoptions(precision=2)          
    plt.figure(figsize=(10,8))
    plot_confusion_matrix(cnf_matrix,classes=class_list, title='Confusion matrix, without normalization')
    plt.show()
    print (set(pred_list))
    print (set(target_list))
    print ('******test_loss:', test_losses/test_x.shape[0],'test_acc:', test_accuracy/test_x.shape[0])
    sess.close()
    return


  ######### TRAINING BELOW ##############
  max_test_acc = 0.
  max_test_id = 0
  sess = tf.Session()
  if run_mode=='train':
    sess.run(tf.global_variables_initializer())
  else:
    saver.restore(sess, checkpoint+model_name)
  model_name = '{}_T{}_state{}'.format(test_subj, sample_rate, hidden_size)
  plot_details = defaultdict(lambda : [])
  total_batches = train_x.shape[0]//batch_size      # ignoring last samples
  for ep in range(nepochs):
    details = defaultdict(lambda: 0.)
    for batch_no in range(total_batches):
      batch_x = train_x[batch_no*batch_size:(batch_no+1)*batch_size,:,:,:,:]
      #batch_x_final = batch_x.reshape(batch_size, sample_rate, -1)
      batch_x_final = batch_x
      batch_y = train_y[batch_no*batch_size:(batch_no+1)*batch_size]
      #_, train_loss, train_acc = sess.run([train_step, total_loss, accuracy], feed_dict={x_ph:batch_x_final, y_ph:batch_y, keep_prob_ph:dropout_keep})
      timg = sess.run(test_image, feed_dict={x_ph:batch_x_final, y_ph:batch_y, keep_prob_ph:dropout_keep})
      print (temp2label[batch_y[1][0]])
      #show_image(timg[0], 'fname')
      #show_image(timg[1], 'fname')
      #show_image(timg[2], 'fname')
      #show_image(timg[3], 'fname')
      show_image(timg[1], 'fname')
      details['ep_loss'] += train_loss
      details['ep_acc'] += train_acc
    print (model_name, 'epochs', ep, 'train_loss:', details['ep_loss']/total_batches, \
    'train_acc:', details['ep_acc']/total_batches, 'global_norm')
    
    if ep and ep%5==0:
      test_details = defaultdict(lambda: 0.)
      for sample_no in range(test_x.shape[0]):
        batch_x = test_x[sample_no:sample_no+1,:,:,:,:]
        #batch_x_final = batch_x.reshape(1, sample_rate, -1)
        batch_x_final = batch_x
        batch_y = test_y[sample_no:sample_no+1]      
        test_loss, test_acc = sess.run([total_loss, accuracy], feed_dict={x_ph:batch_x_final, y_ph:batch_y, keep_prob_ph:1.0})
        test_details['ep_loss'] += test_loss
        test_details['ep_acc'] += test_acc
      print ('******test_loss:', test_details['ep_loss']/test_x.shape[0], \
      'test_acc:', test_details['ep_acc']/test_x.shape[0],'*****', 'max_acc:', max_test_acc, 'max_ep_id', max_test_id)
      details_print()
      if max_test_acc < test_details['ep_acc']/test_x.shape[0]:
        max_test_acc = test_details['ep_acc']/test_x.shape[0]
        max_test_id = ep
        saver.save(sess, '{}/{}_{}'.format(save_folder, model_name,args.model_suffix))      
      plot_details['train_acc'].append(details['ep_acc']/total_batches)
      plot_details['train_loss'].append(details['ep_loss']/total_batches)
      plot_details['test_acc'].append(test_details['ep_acc']/test_x.shape[0])
      plot_details['test_loss'].append(test_details['ep_loss']/test_x.shape[0])
  
  return max_test_acc, test_subj
  

def test_lstm(model_loc=''):  
  return True
def main():
  #make_data(args.test_subj, s_rate =args.sample_rate)
  tensorflow_model(nepochs=args.num_epochs, run_mode=args.run_mode, saved_model_name=args.model_name, \
          sample_rate=args.sample_rate, test_subj=args.test_subj, hidden_size=args.state_size, num_layers=args.layer_no)
  
if __name__ =='__main__':
  main()
