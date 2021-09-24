#!/bin/sh
##
## Job name
#SBATCH --job-name 3dcnn_model_job
## Partition name
#SBATCH --partition gpuq
#SBATCH --gres=gpu:1
## Estimated Time Required in the format D-HH:MM:SS
#SBATCH --time 5-00:00:00
#SBATCH --mem 100000
## %N= node name, %j=job id
#SBATCH -o /scratch/ahosain/slurm_outs/rgb_3dcnn/slurm-%N-%j.out
## Name of error file
#SBATCH -e /scratch/ahosain/slurm_outs/rgb_3dcnn/slurm-%N-%j.err

module load cuda/9.0 python/3.6.7
source /home/ahosain/dl_env/bin/activate

python open_fusion_rnn.py -ts "$ts" -dd /scratch/ahosain/asl_v2_data/openpose_data/deephand_embedding/ -sd "$sd" -ss 1000 -sssk 100 -srsk 20 -ft "$ft" --save_model

#/home/ahosain/dl_env/bin/python option_fusion_rnn.py -ts "$ts" -dd /scratch/ahosain/slurm_outs/rnn_embd_models/embd_sk_data -sd /scratch/ahosain/slurm_outs/rnn_embd_models/full_exp_skaug/saves -ss 4096 -sssk 100 -srsk 20 -ft "$ft" --save_model

# run command for hidden size 1000 for rgb embedding
#python option_fusion_rnn.py -ts "$ts" -dd /scratch/ahosain/slurm_outs/rnn_embd_models/embd_sk_data -sd "$sd" -ss 1000 -sssk 100 -srsk 20 -ft "$ft" --save_model

# run command for alpha pose embedding
# python option_fusion_rnn.py -ts "$ts" -dd /scratch/ahosain/asl_v2_data/deephand_apose_data/ -sd "$sd" -ss 1000 -sssk 100 -srsk 20 -ft "$ft" --save_model

#/home/ahosain/virt_env/bin/python $file_name -sr 15 -rm "$rm" -lr 0.00001 -ne 100  -rb 0.008 -ss 30 -ln 2 -bs 64 -ts "$ts" -mn "$mn"
#/home/ahosain/virt_env/bin/python hello.py 10 "$var1"
