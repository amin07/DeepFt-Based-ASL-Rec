import os
#subs = ['qian','jensen', 'eddie','juan','sofia','fatme','professor']
subs = ['jensen']
fusions = ['only_sk','only_embd','concat','focus_concat']
slurm_loc = '/scratch/ahosain/slurm_outs/openpose_embd_models'
for s in subs:
  for f in fusions:
    os.system("sbatch --output {}/{}_{}.out --error {}/{}_{}.err  --export=ts={},ft={},sd={} run_job.sh".format(slurm_loc, s, f, slurm_loc, s, f, s, f, os.path.join(slurm_loc, 'saves')))
