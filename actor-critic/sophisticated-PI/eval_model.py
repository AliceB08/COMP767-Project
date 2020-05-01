"""
Create ratemaps for a trained network
"""
import torch
from torch import nn
from torch.utils import data
from watermaze_dataloader import Dataset
import glob
import numpy as np
from datetime import datetime
import time
from tqdm import tqdm

from model_lstm import GridTorch, GridTorch_nonNeg
from scores import GridScorer
from utils import *
# from generate_gif import generate_gif

# Parameters
ENV_SIZE = 2.2
RADIUS= 60
BATCH_SIZE = 256 #1024
SEED = 9999
N_PC = [256]
PC_SCALE = [1] # [0.01] [10]
N_HDC = [8]
data_params = {"batch_size": BATCH_SIZE, "shuffle": False, "num_workers": 8}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_file_name(exp, epoch_nb):
    is_file, copy_number = True, 0
    while is_file:
        filename = f"{exp}_epoch{epoch_nb}_{copy_number}.pdf"
        paths = glob.glob(f"./ratemaps/{filename}")
        is_file = len(paths) != 0
        copy_number += 1
    return filename

def create_rate_maps(exp, old_experiment, epoch_nb=None, create_PDF=True, create_gif_frame=False):
    if epoch_nb==None:
        epoch_nb = get_model_epoch(get_latest_model_file(f"./experiments/results/{exp}/"))
    # Loading datasets
    dataset = Dataset(batch_size=data_params["batch_size"])
    data_generator = data.DataLoader(dataset, **data_params)

    # Create the ensembles that provide targets during training
    place_cell_ensembles = get_place_cell_ensembles(env_size=ENV_SIZE, neurons_seed=SEED, n_pc=N_PC)
    head_direction_ensembles = get_head_direction_ensembles(neurons_seed=SEED, n_hdc=N_HDC,)
    target_ensembles = place_cell_ensembles + head_direction_ensembles

    tmp = torch.load(f"./experiments/results/{exp}/target_ensembles.pt")
    place_cell_ensembles[0].means = torch.Tensor(tmp[0].means)
    place_cell_ensembles[0].variances = torch.Tensor(tmp[0].variances)
    head_direction_ensembles[0].means = torch.Tensor(tmp[1].means)
    head_direction_ensembles[0].kappa = torch.Tensor(tmp[1].kappa)

    if old_experiment:
        model = OldGridTorch(target_ensembles, non_linearity=get_exp_non_linearity(exp))
    else:
        model = GridTorch(target_ensembles, non_linearity=get_exp_non_linearity(exp))
    model.load_state_dict(torch.load(f"./experiments/results/{exp}/model_epoch_{epoch_nb}.pt"))
    model.eval()

    for X, y in data_generator:
        break

    # Initialise for forward pass
    init_pos, init_hd, ego_vel = X
    target_pos, target_hd = y
    initial_conds = encode_initial_conditions(init_pos, init_hd, place_cell_ensembles, head_direction_ensembles)
    # Forward pass
    outs = model.forward(ego_vel.transpose(1, 0), initial_conds)
    _, _, bottleneck_acts, _, _ = outs
    acts = bottleneck_acts.transpose(1, 0).detach().numpy()
    pos_xy = target_pos.detach().numpy()

    # Create scorer objects
    starts = [0.2] * 10
    ends = np.linspace(0.4, 1.0, num=10)
    masks_parameters = zip(starts, ends.tolist())
    scorer = GridScorer(20, ((-1.1, 1.1), (-1.1, 1.1)), masks_parameters)

    if create_PDF:
        _ = get_scores_and_plot(scorer, pos_xy, acts, "./ratemaps/", get_file_name(exp, epoch_nb))
    if create_gif_frame:
        generate_gif(scorer, pos_xy, acts, "./ratemaps/", get_file_name(exp, epoch_nb), epoch=epoch_nb)


def create_rate_maps_maze(model_path,target_ensemble_path,exp,non_negativity=False,epoch_nb=None,create_PDF=True,create_gif_frame=False):
    # Load dataset
    dataset = Dataset(batch_size=data_params['batch_size'])
    data_generator = data.DataLoader(dataset, **data_params)

    # Create the target ensembles
    # place_cell_ensembles = get_place_cell_ensembles(env_size=ENV_SIZE, neurons_seed=SEED, n_pc=N_PC)
    place_cell_ensembles = get_place_cell_ensembles(env_size=ENV_SIZE, neurons_seed=SEED, n_pc=N_PC, radial=True, 
        radius=RADIUS,pc_scale=PC_SCALE)
    head_direction_ensembles = get_head_direction_ensembles(neurons_seed=SEED, n_hdc=N_HDC,
        radial=True, radius=RADIUS)
    target_ensembles = place_cell_ensembles + head_direction_ensembles

    # load the saved means from file
    saved_targets = torch.load(target_ensemble_path)
    place_cell_ensembles[0].means = torch.Tensor(saved_targets[0].means)
    place_cell_ensembles[0].variances = torch.Tensor(saved_targets[0].variances)
    head_direction_ensembles[0].means = torch.Tensor(saved_targets[1].means)
    head_direction_ensembles[0].kappa = torch.Tensor(saved_targets[1].kappa)

    # load the saved model
    if not non_negativity:
        model = GridTorch(target_ensembles,n_pcs=N_PC[0],n_hdcs=N_HDC[0])
    else:
        model = GridTorch_nonNeg(target_ensembles,n_pcs=N_PC[0],n_hdcs=N_HDC[0])
    model.load_state_dict(torch.load(model_path,map_location='cpu'))
    model.eval()

    # get first batch of trajectories
    X,y = next(iter(data_generator))
    init_pos, init_hd, ego_vel = X
    target_pos, target_hd = y
    initial_conds, _ = encode_initial_conditions(init_pos, init_hd, place_cell_ensembles, head_direction_ensembles)
    initial_conds = tuple(map(lambda x: x.to(device), initial_conds))
    # do a forward pass
    outs = model.forward(ego_vel.transpose(1, 0), initial_conds)
    _, _, bottleneck_acts, _, _ = outs
    acts = bottleneck_acts.transpose(1, 0).detach().numpy()
    pos_xy = target_pos.detach().numpy()

    # create scorers
    starts = [0.2] * 10
    ends = np.linspace(0.4, 1.0, num=10)
    masks_parameters = zip(starts, ends.tolist())
    scorer = GridScorer(20, ((-60, 60), (-60, 60)), masks_parameters)

    # create plots and gifs
    if create_PDF:
        _ = get_scores_and_plot(scorer, pos_xy, acts, "ratemaps/", get_file_name(exp, epoch_nb),sort_by_score_60=False)
    if create_gif_frame:
        # NOT TESTED!!!
        generate_gif(scorer, pos_xy, acts, "ratemaps/", get_file_name(exp, epoch_nb), epoch=epoch_nb)


if __name__ == "__main__":
    start = time.time()
    # watermaze_experiments = ["2020-04-23_12-06"]
    # EXPERIMENTS = ["2020-04-15_14-40", "2020-04-15_15-25", "2020-04-15_16-21", "2020-04-22_17-06", "2020-04-23_14-23"]
    # Breaking change when using switching layer + target ensemble, specify the old experiments
    # OLD_EXPERIMENTS = ["2020-04-15_14-40", "2020-04-15_15-25", "2020-04-15_16-21", "2020-04-22_17-06", "2020-04-23_14-23"]
    
    '''Uncomment the following lines to generate the PDF'''
    # experiment = EXPERIMENTS[-1]
    # create_rate_maps(experiment, experiment in OLD_EXPERIMENTS)
    # print(f"Done in {time.time()-start:.0f} seconds for batch size {BATCH_SIZE}")

    '''Uncomment the following lines to generate the gif'''
    # experiment = EXPERIMENTS[0]
    # for epoch in tqdm(range(9, 2000, 10)):
    #     create_rate_maps(experiment, experiment in OLD_EXPERIMENTS, epoch_nb=epoch, create_gif_frame=True, create_PDF=False)
    # print(f"Done in {time.time()-start:.0f} seconds for batch size {BATCH_SIZE}")

    ''' Below is the updated plotting for watermaze environment firing rates'''
    model_path = 'exp8/model_epoch_1999.pt'
    target_ensemble_path = 'exp8/target_ensembles.pt'
    create_rate_maps_maze(model_path=model_path, target_ensemble_path=target_ensemble_path, non_negativity=False, exp='exp8_90',create_PDF=True)
    print(f"Done in {time.time()-start:.0f} seconds for batch size {BATCH_SIZE}")    

