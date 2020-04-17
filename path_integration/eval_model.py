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

from model_lstm import GridTorch
from model_utils import get_latest_model_file, get_model_epoch
from scores import GridScorer
from utils import *

# Parameters
ENV_SIZE = 2.2
BATCH_SIZE = 4000
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

def create_rate_maps(exp, epoch_nb=None):
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

    model = GridTorch(target_ensembles)
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

    scoress = get_scores_and_plot(scorer, pos_xy, acts, "./ratemaps/", get_file_name(exp, epoch_nb))


if __name__ == "__main__":
    EXPERIMENTS = ["2020-04-15_14-40", "2020-04-15_15-25", "2020-04-15_16-21"]
    for i in [1,2]:
        create_rate_maps(EXPERIMENTS[i])