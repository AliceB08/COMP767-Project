"""
Create ratemaps for a trained network
"""
import torch
from torch import nn
from torch.utils import data
from dataloading import Dataset
import glob
import numpy as np
from datetime import datetime
import time
from tqdm import tqdm

from model_lstm import GridTorch
from old_model_to_use_for_plots.model_lstm import GridTorch as OldGridTorch
from model_utils import *
from scores import GridScorer
from utils import *
from generate_gif import generate_gif

# Parameters
ENV_SIZE = 2.2
BATCH_SIZE = 1024
SEED = 9999
N_PC = [256]
N_HDC = [12]
data_params = {"batch_size": BATCH_SIZE, "shuffle": False, "num_workers": 8}


def get_file_name(exp, epoch_nb):
    is_file, copy_number = True, 0
    while is_file:
        filename = f"{exp}_epoch{epoch_nb}_{copy_number}.pdf"
        paths = glob.glob(f"./ratemaps/{filename}")
        is_file = len(paths) != 0
        copy_number += 1
    return filename


def create_rate_maps(exp, old_experiment, epoch_nb=None, create_PDF=True, create_gif_frame=False, target_set_nb=0, sort_by_score=60):
    if epoch_nb == None:
        epoch_nb = get_model_epoch(get_latest_model_file(f"./experiments/{exp}/"))
    # Loading datasets
    dataset = Dataset(batch_size=data_params["batch_size"])
    data_generator = data.DataLoader(dataset, **data_params)

    # Create the ensembles that provide targets during training

    tmp = torch.load(f"./experiments/{exp}/target_ensembles.pt")

    if old_experiment:
        place_cell_ensembles = get_place_cell_ensembles(env_size=ENV_SIZE, neurons_seed=SEED, n_pc=N_PC)
        head_direction_ensembles = get_head_direction_ensembles(neurons_seed=SEED, n_hdc=N_HDC,)
        target_ensembles = place_cell_ensembles + head_direction_ensembles
        place_cell_ensembles[0].means = torch.Tensor(tmp[0].means)
        place_cell_ensembles[0].variances = torch.Tensor(tmp[0].variances)
        head_direction_ensembles[0].means = torch.Tensor(tmp[1].means)
        head_direction_ensembles[0].kappa = torch.Tensor(tmp[1].kappa)
        model = OldGridTorch(target_ensembles, non_linearity=get_exp_non_linearity(exp))
        model.load_state_dict(torch.load(f"./experiments/{exp}/model_epoch_{epoch_nb}.pt"))
    else:
        N_SWITCHING_TARGETS = len(tmp)
        target_ensembles = []
        for i in range(N_SWITCHING_TARGETS):
            place_cell_ensembles = get_place_cell_ensembles(env_size=ENV_SIZE, neurons_seed=SEED, n_pc=N_PC)
            head_direction_ensembles = get_head_direction_ensembles(neurons_seed=SEED, n_hdc=N_HDC)
            target_ensembles.append(place_cell_ensembles + head_direction_ensembles)
        for i in range(N_SWITCHING_TARGETS):
            target_ensembles[i][0].means = torch.Tensor(tmp[i][0].means)
            target_ensembles[i][0].variances = torch.Tensor(tmp[i][0].variances)
            target_ensembles[i][1].means = torch.Tensor(tmp[i][1].means)
            target_ensembles[i][1].kappa = torch.Tensor(tmp[i][1].kappa)
        model = GridTorch(
            target_ensembles, non_linearity=get_exp_non_linearity(exp), n_switching_targets=N_SWITCHING_TARGETS
        )
        model.load_state_dict(torch.load(f"./experiments/{exp}/model_epoch_{epoch_nb}.pt"))

    model.eval()

    for X, y in data_generator:
        break

    # Initialise for forward pass
    init_pos, init_hd, ego_vel = X
    target_pos, target_hd = y
    # Forward pass
    if old_experiment:
        initial_conds = encode_initial_conditions(init_pos, init_hd, place_cell_ensembles, head_direction_ensembles)
        outs = model.forward(ego_vel.transpose(1, 0), initial_conds)
    else:
        initial_conds = encode_initial_conditions(
            init_pos, init_hd, [target_ensembles[target_set_nb][0]], [target_ensembles[target_set_nb][1]]
        )
        outs = model.forward(ego_vel.transpose(1, 0), initial_conds, target_set_nb)
    _, _, bottleneck_acts, _, _ = outs
    acts = bottleneck_acts.transpose(1, 0).detach().numpy()
    pos_xy = target_pos.detach().numpy()

    # Create scorer objects
    starts = [0.2] * 10
    ends = np.linspace(0.4, 1.0, num=10)
    masks_parameters = zip(starts, ends.tolist())
    scorer = GridScorer(20, ((-1.1, 1.1), (-1.1, 1.1)), masks_parameters)

    if create_PDF:
        _ = get_scores_and_plot(scorer, pos_xy, acts, f"./experiments/{exp}/", get_file_name(exp, epoch_nb), sort_by_score=sort_by_score)
    if create_gif_frame:
        generate_gif(scorer, pos_xy, acts, f"./experiments/{exp}/", get_file_name(exp, epoch_nb), epoch=epoch_nb)


if __name__ == "__main__":
    start = time.time()
    '''
    List of experiments:
    1 - Original
    2 - PadCoder
    3 - Dropout=0.5
    4 - Transfer Learning
    5 - tanh
    6 - ReLU
    7 - HDC
    8 - PC
    New architecture:
    9 - Switchin Heads
    10 - Agnostic hypothesis
    11 - Second ReLU
    '''

    """Uncomment the following lines to generate the PDF"""
    experiment_nb = 1
    experiment_folder = f"PI_EXP_{experiment_nb}"
    create_rate_maps(experiment_folder, experiment_nb<9, sort_by_score=60)
    print(f"Done in {time.time()-start:.0f} seconds for batch size {BATCH_SIZE}")

    """Uncomment the following lines to generate the gif"""
    # experiment_nb = 1
    # experiment_folder = f"PI_EXP_{experiment_nb}"
    # for epoch in tqdm(range(9, 2000, 10)):
    #     create_rate_maps(experiment_folder, experiment_nb<9, epoch_nb=epoch, create_gif_frame=True, create_PDF=False)
    # print(f"Done in {time.time()-start:.0f} seconds for batch size {BATCH_SIZE}")
