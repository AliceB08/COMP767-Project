# Description of files, data & results in sophisticated PI folder
## Environment files
* __watermaze.py__ --> describes the env and tasks

## Path integration files
Files noted in order of workflow/pipeline:

* __watermaze_dataloader.py__ --> pytorch dataloader for loading the trajectory dataset collected from watermaze
* __ensembles.py__ -->  Generates the place and HD cell activations for the trajectory points, needed for the model training
* __model_lstm.py__ --> Describes the LSTM models (with and without non-negative activation, i.e. ReLU). Possibly should be one class instead of two classes
* __model_utils.py__ --> Defines functions to resume training from some epoch/stage
* __scores.py__ --> Defines the ortho/hexa symmetry calculation functions + SAC and rate map calculation and plotting functions
* __utils.py__ --> Describes the important functions for encoding inputs, initial conditions and plotting the correlation scores + SAC
* __train_wmaze_path_integration.py__ --> The main path integration training file, all hyperparams to be set here and *this should be run* to train PI
* __run_train.sh__ --> The script file to run train_wmaze_path_integration on MILA cluster *(change the virtualenv name)*
* __eval_model.py__ --> Creates rate maps and SAC plots and also creates pdfs ordered by hexa/ortho symmetry
* __eval_model_performance_trajectory.py__ --> Plots the ground truth and predicted trajectory

### What to run
1. Train model by running train_wmaze_path_integration.py or run_train.sh
2. Use the trained model to generate ratemaps by running eval_model.py. Change *sort_by_score_60=True* to sort by hexagonal scores (False for orthogonal scores). Change the fname as well
3. [If you want to visualize the accuracy of position encoding] Run eval_model_performance_trajectory.py to plot the discrepancies between GT and predicted trajectories

__Note:__ Do not forget to set the hyperparams in each file. The hyperparams should be the same as the ones the model was trained using

## Actor-critic RL files
Files noted in order of workflow/pipeline:

* __placeCellEncoding.py__ --> Defines a place cell encoder that uses some defined number of place cells (Gaussian place field with defined sigma)
* __asymmetricTileCoding.py__ --> Defines a tile coder to discretize the environment into defined number of tilings and bins
* __gridCellEncoding.py__ --> Defines a grid coder using one of the pretrained path integration modules
* __actorCritic_goalCode.py__ --> Defines the linear actor-critic agent that uses one of the 3 feature reprsentations and has the functionality of goal memory

### What to run
1. Run the actorCritic_goalCode.py in order to get escape latency plots for any of the watermaze tasks. Take note of the parameters required by the AC agent object.

-------------------------------------------------------------------
## Results
1. ratemaps and SAC plots are in [__ratemaps__](sophisticated%20PI/ratemaps/) folder
2. RL plots are in the [__navigation result plots__](navigation%20result%20plots/) folder
-------------------------------------------------------------------
## Data
* [__exp1__](sophisticated%20PI/exp1/) --> contains the model from Mandana's experiment (no constraint and PC stdev=0.01) --> orthogonal (~1.5)
* [__exp2__](sophisticated%20PI/exp2/) --> contains the model from 2nd experiment (no constraint and PC stdev=10) --> weaker orthogonal (~1.2)
* [__exp3__](sophisticated%20PI/exp3/) --> contains the model from 3rd experiment (non-negativity constraint and PC stdev=10) --> still orthogonal (~1.26)
