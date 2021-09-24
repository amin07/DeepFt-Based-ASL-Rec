import sys
import os
from glob import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_loc", help="from where to copy dta/fldr", type=str, required=True)
parser.add_argument("--out_loc", help="to where to copy", type=str, required=True)
args = parser.parse_args()

subs_order = "alamin	paneer	kaleab	ding	qian	jensen	eddie	aiswarya	juan	sofia	fatme	professor".split()
subs_id = ["subject{:02d}".format(i) for i in range(1, 13)]
name_map = dict([(t[1], t[0]) for t in zip(subs_id, subs_order)])


ctr = 0
for dirs in os.walk(args.input_loc):
  folder_loc, _, files = dirs
  for each_file in files:
    old_path = os.path.join(folder_loc, each_file)
    for sub in subs_order:
      occ_idx = old_path.find(sub)
      if occ_idx>=0:
        new_path = os.path.join(args.out_loc, old_path[occ_idx:])
        new_path = new_path.replace(sub+'_', name_map[sub]+'_')   
        new_path = new_path.replace('_'+sub+'_', '_'+name_map[sub]+'_')
        new_path = new_path.replace('_'+sub, '_'+name_map[sub])
        if os.path.exists(os.path.dirname(new_path))==False:
          os.makedirs(os.path.dirname(new_path))
        comm = "cp {} {}".format(old_path, new_path)
        os.system(comm)
        print ('written', ctr+1)
        ctr += 1
