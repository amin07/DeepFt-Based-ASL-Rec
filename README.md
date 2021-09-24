# Deep Hand Feature Based American Sign Language (ASL) Recognition.
Learning ASL video modeling using deep hand-patch features and pose (skeleton) based features <br>
Here is the diagram of the overall architecture, <br><br>
<img src="repo_imgs/arch.JPG" width="600" height="400" />
<br><br><br>

This repository contains source code for the following paper,

A. A. Hosain, P. Selvam Santhalingam, P. Pathak, J. Košecké and H. Rangwala, **"Body Pose and Deep Hand-shape Feature Based American Sign Language Recognition,"** 
2020 IEEE 7th International Conference on Data Science and Advanced Analytics (DSAA), 2020, pp. 207-215, doi: 10.1109/DSAA49011.2020.00033.


## Paper/Cite
```
@INPROCEEDINGS{9260043,
  author={Hosain, Al Amin and Selvam Santhalingam, Panneer and Pathak, Parth and Košecké, Jana and Rangwala, Huzefa},
  booktitle={2020 IEEE 7th International Conference on Data Science and Advanced Analytics (DSAA)}, 
  title={Body Pose and Deep Hand-shape Feature Based American Sign Language Recognition}, 
  year={2020},
  volume={},
  number={},
  pages={207-215},
  doi={10.1109/DSAA49011.2020.00033}}
```
## Data Download
* The network is trained on two types of input representation of an ASL video: the rgb hand feature and the pose data
* The data download link: [hand feature and pose download](https://drive.google.com/file/d/1BBwRGU8W17TK_eU_28y51c1Q-O7HqUmU/view?usp=sharing)

## Set up running environment
Run following commands after cloning the repo,
```
conda env create -f environments.yml
conda activate finehand_env
```

## Network 1 : Hand shape network
This is a image recognition type convolutional neural netowrk (CNN). An instance of ResNet50 was used here. Any other compatible CNN can be used. The goal of this CNN is to learn hand shape patterns as shown below. The per frame learned representation will be used later in the sign recognition phase.

Explanation of the options

* -hdir: directory of the hand shape obtained in each of the iterations (contains three iterations for our experiment)
* --test_subj: the subject for which we ignore all the training hand shape image because, this is the test subject
* --save_model: bool option, if specified, the script will save the trained cnn based on eval accuracy on test hand patches
* -ct: train type using both hands or single hand options are [both_hand, left_hand, right_hand]
* -lr: learning rate

### Training 
* ```python run_handshape_model.py -hdir <hand_shape_image_directory>  --test_subj <subject identifier> --save_model```
* **An example run:** ```python run_handshape_model.py -hdir data_iters/iter2/ -ts subject03```
### Evaluation
* ```python run_handshape_model.py -hdir <hand_shape_image_directory>  --test_subj subject03 -rm test -tm <full_location_to_trained_model>```
* **Example run:** ```python run_handshape_model.py -hdir  data_iters/iter2/ -ts subject03 -tm saves/handshape-model-subject03 -rm test```


## Network 2 : Recurrent Sign recognition network

In this training, the input hand-patch videos are used to extract hand features using the trained model in the previous step. The all hand feature embeddings are first save into temporary direcotry and the sign recognition models are trained on those embeddings. Finally, the temporary directory is being cleaned.

### Training and Evaluation
The lstm based sign recognition model will be trained and show maximum test accuracy on following commands,
* ```python run_lstm_sign_model.py -hcnn <saved_handshape_model_location> -dd <cropped_hand_video_direcotry> -bs <batch_size> -sr <sample_rate> -lr <learning_rate> -ts <test_subject> -ct <both hand vs single hand>```
* **An example run:** ```python run_lstm_sign_model.py -hcnn saves/handshape-model-subject03 -dd cropped_handpatches/ -bs 8 -sr 20 -lr 0.0001 -ts subject03 -ct both_hand```


