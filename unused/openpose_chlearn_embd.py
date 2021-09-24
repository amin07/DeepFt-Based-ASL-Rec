import csv
import sys
import os
import numpy as np
import cv2
import random
from random import shuffle
import matplotlib.pyplot as plt
from collections import defaultdict
import glob
import json
from random import shuffle
from collections import defaultdict
from image_embedding import get_embeddings
np.set_printoptions(suppress=True)

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
vpath = '/scratch/ahosain/asl_v2_data/segmented_data/'
def get_hand_patches(vid_file, json_loc, w_size = (100, 100)):
  vid_name = os.path.basename(vid_file)
  vid_name = vid_name.split('.')[0]
  vid = cv2.VideoCapture(vid_file)
  fcount = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
  #print (os.path.join(json_loc, vid_name+'*'))
  json_files = sorted(glob.glob(os.path.join(json_loc, vid_name+'*')))
  if len(json_files)!=fcount:
    print ('frame count mismatch. do something!', len(json_files), fcount)
    sys.exit()

  motion_ar = []
  first_frame = (None, None)
  frame_ar = []
  pose_ar = []
  for i, jf in enumerate(json_files):
    ret, frame = vid.read()
    frame_ar.append(frame.copy())
    with open(jf, 'r') as f:
      json_dat = json.load(f)['people'][0]    # top dict has keys: version, people
      pose_kps = np.split(np.array(json_dat['pose_keypoints_2d']), 25)
      lhand_kps = np.split(np.array(json_dat['hand_left_keypoints_2d']), 21)
      rhand_kps = np.split(np.array(json_dat['hand_right_keypoints_2d']), 21)
      lhand_med = sum(lhand_kps)/len(lhand_kps)
      rhand_med = sum(rhand_kps)/len(rhand_kps)
      pose_ar.append((pose_kps, lhand_kps, rhand_kps))
      '''
      for jp  in pose_kps:
        joint_x, joint_y = int(jp[0]), int(jp[1])
        cv2.circle(frame, (joint_x, joint_y), 10, (255,0,255), -1)
      for jp  in lhand_kps+rhand_kps:
        joint_x, joint_y = int(jp[0]), int(jp[1])
        cv2.circle(frame, (joint_x, joint_y), 4, (255,0,2), -1)
      cv2.circle(frame, (int(lhand_med[0]), int(lhand_med[1])), 6, (25,250,2), -1)
      cv2.circle(frame, (int(rhand_med[0]), int(rhand_med[1])), 6, (25,250,2), -1)
      '''
      #frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)        
      '''
      cv2.imshow('frames', frame)
      cv2.waitKey(-1)
      '''
      # calculatin motion
      if i==0:
        first_frame = (lhand_med, rhand_med)
        motion_ar.append(0.)
      else:
        curr_dist = sum([(t[0]-t[1])**2 for t in zip(first_frame[0][:2], lhand_med[:2])])
        curr_dist += sum([(t[0]-t[1])**2 for t in zip(first_frame[1][:2], rhand_med[:2])])
        motion_ar.append(curr_dist)

  th = 0.2
  avg_ar = [sum(motion_ar)/len(motion_ar)]*len(motion_ar)
  avg_th = [v*th for v in avg_ar]

  '''
  plt.plot(motion_ar)
  plt.plot(avg_ar)
  plt.plot(avg_th)
  plt.show()
  '''
  th_motion = (sum(motion_ar)/len(motion_ar))*th
  start_pos, end_pos = None, None
  gap_list = []
  for i, v in enumerate(motion_ar):
    if v > th_motion and start_pos==None:
      start_pos = i
    if start_pos and v<th_motion:
      end_pos = i-1
      gap_list.append((start_pos, end_pos))
      start_pos = None
      end_pos = None 
  
  if start_pos and not end_pos:
    gap_list.append((start_pos, len(motion_ar)-1))

  gaps = [ts[1]-ts[0] for ts in gap_list]
  start_pos, end_pos = gap_list[np.argmax(gaps)]
  print (start_pos, end_pos, 'se pos')
  def crop_hand_patches(vid, sk_dat, w_size = (100, 100)):
    left_hand, right_hand = [], []
    fcount  = len(sk_dat)
    for i in range(fcount):
      frame = vid[i]
      try:
        cw, ch = w_size
        body_kps, lhand_kps, rhand_kps = sk_dat[i]
        lhand_med = sum(lhand_kps)/len(lhand_kps)
        rhand_med = sum(rhand_kps)/len(rhand_kps)
        joint_x, joint_y = int(lhand_med[0]), int(lhand_med[1])
        sx, sy = max(0, joint_x - ch//2), max(0, joint_y - cw//2)
        left_hand.append(frame[sy:sy+cw,sx:sx+ch,:])
        joint_x, joint_y = int(rhand_med[0]), int(rhand_med[1])
        sx, sy = max(0, joint_x - ch//2), max(0, joint_y - cw//2)
        right_hand.append(frame[sy:sy+cw,sx:sx+ch,:])
      except Exception as e:
        print ('exception cropping',e)
    return left_hand, right_hand
  
  frame_ar = frame_ar[start_pos:end_pos]
  pose_ar = pose_ar[start_pos:end_pos]
  lhands, rhands = crop_hand_patches(frame_ar, pose_ar)
  #lhands, rhands = np.array(lhands), np.array(rhands)
  #print (lhands.shape, rhands.shape)
  return np.concatenate((lhands, rhands), axis=1), pose_ar 
   

vid_loc = sys.argv[1]
out_loc = sys.argv[2]

vid_files = sorted([ v for v in os.listdir(vid_loc) if v.split('_')[-1]=='color.mp4'])
vid_files = vid_files[:]
hand_list = []
file_list = []
bad_cnt = 0
#total_file = 0
for i, vf in enumerate(vid_files):
  vid_file = os.path.join(vid_loc, vf)
  vid_name = os.path.basename(vid_file).split('.')[0]
  sk_file = os.path.join(vid_loc, vid_name.split('_')[0]+'_skeleton.csv') 
  label_file = os.path.join(vid_loc, vid_name.split('_')[0]+'_labels.csv') 
  #if not os.path.exists(label_file) or not os.path.exists(sk_file):
  #  print ('label or sk missing')
  #  continue
  
  print (vid_file, vid_name, sk_file, os.path.isfile(sk_file))
  class_no = [0]*20  
  with open(label_file) as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    csvfile_sk = open(sk_file)
    readSK = csv.reader(csvfile_sk, delimiter=',')

    vid = cv2.VideoCapture(vid_file)
    fcount = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    fwidth = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    fheight = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    curr_frame_no = 1
    file_list = []
    hand_list = []
    for row in readCSV:
      class_label, start_pos, end_pos = int(row[0]), int(row[1]), int(row[2])
      class_no[class_label-1] += 1
      while curr_frame_no < start_pos:
        _, frame = vid.read()
        line = next(readSK)
        curr_frame_no += 1

      #fourcc = cv2.VideoWriter_fourcc(*'XVID')
      #vid_name = os.path.join(out_loc, '{}_{}.avi'.format(sample_name, class_label))
      #print (vid_name)
      #vid_writer = cv2.VideoWriter(vid_name, fourcc, 10.0, (fwidth, fheight), True)
      sample_frames = []
      lhands, rhands = [], []
      pose_list = []
      for i in range(start_pos, end_pos+1):
        sk_frame = np.split(np.array(next(readSK)), 20)
        pose_list.append(sk_frame)
        _, frame = vid.read()

        cw, ch = 100, 100
        left_wrist = sk_frame[7][-2:]
        right_wrist = sk_frame[11][-2:]

        #print (left_wrist, right_wrist)
        joint_x = int(left_wrist[0])
        joint_y = int(left_wrist[1])
        sx, sy = max(0, joint_x - ch//2), max(0, joint_y - cw//2)
        if sx+ch >= fwidth:         # cause we using sx as width and sy as height
          sx -= (sx+ch - fwidth)
        if sy+cw >= fheight:
          sy -= (sy+cw - fheight)

        lhands.append(frame[sy:sy+cw,sx:sx+ch,:])
        #cv2.imshow('frame', cropped_frame) 
        #cv2.waitKey(-1)
     
        joint_x = int(right_wrist[0])
        joint_y = int(right_wrist[1])
        sx, sy = max(0, joint_x - ch//2), max(0, joint_y - cw//2)
        if sx+ch >= fwidth:
          sx -= (sx+ch - fwidth)
        if sy+cw >= fheight:
          sy -= (sy+cw - fheight)
        rhands.append(frame[sy:sy+cw,sx:sx+ch,:])
        #cv2.imshow('frame', cropped_frame) 
        #cv2.waitKey(-1)
        #vid_writer.write(frame)
        curr_frame_no += 1
      #np.save(os.path.join(), ) 
      #print (len(lhands), len(rhands), lhands[0].shape)
      try:
        if os.path.exists(os.path.join(out_loc, vid_name.split('_')[0]+'_{}_{}_embd.npy'.format(class_no[class_label-1], class_label))):
          print ('Done')
          break
        np.save(os.path.join(out_loc, vid_name.split('_')[0]+'_{}_{}_pose'.format(class_no[class_label-1],class_label)), pose_list)
        hand_list.append(np.concatenate((lhands, rhands), axis=1))
        file_list.append(vid_name.split('_')[0]+'_{}_{}_embd'.format(class_no[class_label-1], class_label))
      except:
        print ('bad files')
        bad_cnt += 1
      #print (hand_list[-1].shape)
      #print (file_list[-1])
      #vid_writer.release()
  if len(hand_list)==0:
    continue
  print (len(hand_list), len(file_list))
  get_embeddings(hand_list, file_list, out_loc=out_loc)

#print ('total_file', total_file)
print ('total_bad', bad_cnt)
'''
vid_loc = sys.argv[1]
json_loc = sys.argv[2]
out_loc = sys.argv[3]

vid_files = [ v for v in os.listdir(vid_loc) if v.split('.')[-1]=='avi']
vid_files = vid_files[:]
hand_list = []
file_list = []
bad_cnt = 0
for i, vf in enumerate(vid_files):
  vid_file = os.path.join(vid_loc, vf)
  vid_name = os.path.basename(vid_file).split('.')[0]
  print (vid_file)
  try:
    hands, poses = get_hand_patches(vid_file, json_loc)
    print (i+1, 'cropped ',hands.shape, len(poses))
  except Exception as e:
    print ('bad files', e)
    bad_cnt += 1
    continue
  hand_list.append(hands)
  file_list.append(vid_name+'_embd')
  np.save(os.path.join(out_loc, vid_name[:-4]+'_bodyPose'), [t[0] for t in poses])
  np.save(os.path.join(out_loc, vid_name[:-4]+'_leftHandPose'), [t[1] for t in poses])
  np.save(os.path.join(out_loc, vid_name[:-4]+'_rightHandPose'), [t[2] for t in poses])
get_embeddings(hand_list, file_list, out_loc=out_loc)
  # write poses here
print ('total bad', bad_cnt)
'''
