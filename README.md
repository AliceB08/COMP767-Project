# Reinforcement Learning Final Project - COMP767

## Getting started

Clone the repository `git clone https://github.com/AliceB08/COMP767-Project.git`.

Download the Pytorch data for Path Integration from the Google Drive: https://drive.google.com/drive/folders/15_QpZHuSnTMR_OIRehqJfPo68O1twfAd?usp=sharing. This data folder should be placed at the same level as this README,path_integration and actor-critic folders.

## Requirements

- Pytoch
- numpy
- matplotlib
- scipy

## Folders

The `path_integration` folder contains the Pytorch implementation of the paper of the supervised training for the Grid Cell network. The original repo from the paper (TensorFlow implementation) can be found here: https://github.com/deepmind/grid-cells. A repository with a PyTorch adaptation can be found here: https://github.com/LPompe/gridtorch.

## Run the Path Integration experiments

Once in the `path_integration` folder, you can execute the bash files in the `training_scripts` folder:
- `run_train.sh`: the basic experiment with default parameters
- `change_activation.sh`: changes the activation from non to 'relu' or 'tanh' as an option
- `run_test_pretrained_lstm.sh`: loads the pretrained LSTM from the last model in a folder given as argument, then generates new targets with a given seed and only trains the last layers on the trajectories.
- `run_train_from_saved.sh`: loads a pre-trained model (all the model, not only the LSTM)
- `run_train_switching_targets.sh`: generates multiple target ensembles and the same number of last layers. Trains the model with switching targets.

The weights and ratemaps for the path integration experiments are in the folder `experiments`.

## Actor-Critic

The necessary information to run the actor-critic is in the README in the `actor-critic` directory. See https://github.com/AliceB08/COMP767-Project/blob/master/actor-critic/README.md.