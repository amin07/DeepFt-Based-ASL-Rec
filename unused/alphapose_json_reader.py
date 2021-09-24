import json
import numpy as np
import cv2
import os
import sys
from random import shuffle
from collections import defaultdict
np.set_printoptions(suppress=True)
fpath = '/scratch/ahosain/AlphaPose/AlphaPose/examples/res/'
vpath = '/scratch/ahosain/asl_v2_data/segmented_data/'
out_loc = '/scratch/ahosain/asl_v2_data/deephand_apose_data'

'''
vid : rgb video file reader
sk_dat : a list where each element reps sk data of a frame (alphapose default format-COCO)
'''
def crop_hand_patches(vid, sk_dat, w_size = (224, 224)):
  left_hand, right_hand = [], []
  fcount  = len(sk_dat)

  for i in range(fcount):
    _, frame = vid.read()
    try:
      cw, ch = w_size
      joint_details = sk_dat[i]
      
      joint_x = int(joint_details[9][0])      # index 9 ==> left wrist joint
      joint_y = int(joint_details[9][1])
      sx, sy = max(0, joint_x - ch//2), max(0, joint_y - cw//2)
      left_hand.append(frame[sy:sy+cw,sx:sx+ch,:])
      joint_x = int(joint_details[10][0])
      joint_y = int(joint_details[10][1])
      sx, sy = max(0, joint_x - ch//2), max(0, joint_y - cw//2)
      right_hand.append(frame[sy:sy+cw,sx:sx+ch,:])
    except Exception as e:
      print ('exception cropping',e)
  return left_hand, right_hand


sub_names = [ f for f in (os.listdir(vpath)) if f not in ['userA', 'userB']]
print (sub_names)
mismatch = 0
crop_w = (244, 244)     # acc to deephand
for sub in sub_names:
  pose_loc = os.path.join(fpath, sub)
  vid_loc = os.path.join(vpath, sub)
  vid_files = [f for f in os.listdir(vid_loc) if f.split('.')[1]=='avi']
  croph, cropw = crop_w
  shuffle(vid_files)
  for v in vid_files[:10]:
    vid_file = os.path.join(vid_loc, v)
    pose_file = os.path.join(pose_loc, v.split('.')[0]+'_json.json')

    with open(pose_file, 'r') as f:
      data = json.load(f)
      pose_frame_cnt = len(data)
    
    frame_dict = defaultdict(lambda : [])
    for fr in data:
      frame_dict[fr['image_id']].append(fr)
    pure_data = []
    for k in frame_dict.keys():
      if len(frame_dict[k])==1:
        pure_data.append(frame_dict[k][0])
      else:
        max_conf_idx = np.argmax([e['score'] for e in frame_dict[k]])
        pure_data.append(frame_dict[k][max_conf_idx])
    
    pose_frame_cnt = len(pure_data)
  
    vid = cv2.VideoCapture(vid_file)
    vid_frame_cnt = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    
    sk_dat = []
    for fdict in pure_data:
      kps = np.split(np.array(fdict['keypoints']), 17)      # 17 keypoints
      sk_dat.append(kps)
 
    lh_patches, rh_patches = crop_hand_patches(vid, sk_dat, crop_w)
    print (len(lh_patches), len(rh_patches))
    print (lh_patches[0].shape)
    
    # writing the hand patch video for testing purpose
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    vid_writer = cv2.VideoWriter(os.path.join(out_loc, os.path.basename(vid_file)), fourcc, 10.0, (2*cropw,croph), True)
    for pts in zip(lh_patches, rh_patches):
      conc = np.concatenate((pts[0], pts[1]), axis=1)
      vid_writer.write(conc)
    vid_writer.release()
    if pose_frame_cnt!=vid_frame_cnt:
      print (vid_file, os.path.exists(vid_file), vid_frame_cnt)
      print (pose_file, os.path.exists(pose_file), pose_frame_cnt)
      mismatch += 1

print (mismatch)
      
