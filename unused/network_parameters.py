import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--save_dir", "-sd", help="directory where models to be saved", type=str, default='/scratch/ahosain/slurm_outs/rnn_embd_models/saves')
parser.add_argument("--test_model", "-tm", help="model to be tested.",type=str, default='')
parser.add_argument("--fusion_type", "-ft", help="type of fusion.",type=str, default='concat')
parser.add_argument("--test_subj","-ts",  help="name or id of the test subject.",type=str, default=None)
parser.add_argument("--data_dir", "-dd", help="location of input embeddings. default to original embd.",type=str, default='/home/ahosain/asl_rgb_embedding/deephand/TF-DeepHand/embeddings')
parser.add_argument("--input_len","-il",  help="len of input dimension. default is embedding size 2*1024",type=int, default=2*1024)
parser.add_argument("--state_size","-ss",  help="state size of lstm. default double of input len.",type=int, default=None)
parser.add_argument("--state_size_sk","-sssk",  help="state size of lstm. default double of input len. sk",type=int, default=None)
parser.add_argument("--num_layers","-nl",  help="number of layers.",type=int, default=2)
parser.add_argument("--num_layers_sk","-nlsk",  help="number of layers sk lstm.",type=int, default=2)
parser.add_argument("--sample_rate","-sr",  help="frames considered per sample.",type=int, default=15)
parser.add_argument("--sample_rate_sk","-srsk",  help="frames considered per sample sk.",type=int, default=15)
parser.add_argument("--num_epochs","-ne",  help="number of epochs.",type=int, default=50)
parser.add_argument("--batch_size","-bs",  help="size of each batch.",type=int, default=64)
parser.add_argument("--drop_out","-do",  help="drop out keep probability.",type=float, default=0.5)
parser.add_argument("--learning_rate","-le",  help="learning rate.",type=float, default=0.0001)
parser.add_argument("--save_model",  help="if save model or not.", action='store_true')
parser.add_argument("--concat_score","-cs",  help="concat score of concatenation.",type=float, default=0.2)

args = parser.parse_args()

if not args.test_subj and not args.test_model:
  print ('Error: no test subject specified in training!')
elif not args.test_subj:
  args.test_subj = args.test_model.split('_')[0]

if args.state_size==None:
  args.state_size = args.input_len*2

subs = ['paneer','kaleab','juan', 'qian','alamin', 'aiswarya', 'professor','eddie', 'jensen', 'ding', 'sofia', 'fatme']
train_subs = [s for s in subs if s!=args.test_subj]
test_subs = [args.test_subj]

