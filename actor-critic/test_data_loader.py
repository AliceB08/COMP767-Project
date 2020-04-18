import torch
from torch.utils import data
from torch import nn
import numpy as np
from tqdm import tqdm
import time
import argparse
from watermaze-dataloader import Dataset
from model_utils import get_latest_model_file, get_model_epoch
from model_lstm import GridTorch
from utils import *
import os
import time

# Arguments
parser = argparse.ArgumentParser(description="PyTorch Grid Cells Path Integration")
parser.add_argument("--env_size", type=int, default=2.2, help="size of environment")
parser.add_argument("--num_epochs", type=int, default=10, help="number of epochs to stop after")
parser.add_argument("--steps", type=int, default=100, help="steps per epoch")
parser.add_argument("--batch_size", type=int, default=128, help="size of one minibatch")
parser.add_argument("--grad_clip", type=float, default=1e-5, help="gradient clipping")
parser.add_argument("--num_place_cells", type=int, default=256, help="number of place cells")
parser.add_argument("--num_headD_cells", type=int, default=12, help="number of head direction cells")
parser.add_argument("--btln_dropout", type=float, default=0.5, help="bottleneck dropout")
parser.add_argument("--weight_decay", type=float, default=1e-5, help="weight decay")
parser.add_argument("--lr", type=float, default=1e-5, help="initial learning rate")
parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
parser.add_argument("--hidden_size", type=int, default=256, help="size of hidden layers")
parser.add_argument(
    "--save_dir",
    type=str,
    default="./experiments/results/",
    help="path to save the experimental config, logs, model. This is not automatically generated.",
)
parser.add_argument("--use_saved_model", type=bool, default=False, help="Use previously trained model")
parser.add_argument("--seed", type=int, default=9999, help="random seed")
parser.add_argument("--save_model_freq", type=int, default=10, help="Save model every X epochs")
args = parser.parse_args()
argsdict = args.__dict__

# Setting up cuda and seeds
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.manual_seed(argsdict["seed"])
np.random.seed(argsdict["seed"])

# Parameters
N_PC = [argsdict["num_place_cells"]]
N_HDC = [argsdict["num_headD_cells"]]
data_params = {"batch_size": argsdict["batch_size"], "shuffle": True, "num_workers": 6}
test_params = {"batch_size": 100, "shuffle": True, "num_workers": 2}
# coder = PadCoder(3)
coder = None

# Loading datasets
dataset = Dataset(batch_size=data_params["batch_size"])
data_generator = data.DataLoader(dataset, **data_params)
test_generator = data.DataLoader(dataset, **test_params)

# Create the ensembles that provide targets during training
place_cell_ensembles = get_place_cell_ensembles(env_size=argsdict["env_size"], neurons_seed=argsdict["seed"], n_pc=N_PC)
head_direction_ensembles = get_head_direction_ensembles(neurons_seed=argsdict["seed"], n_hdc=N_HDC)
target_ensembles = place_cell_ensembles + head_direction_ensembles

# Create model and restore previous model if desired
model = GridTorch(target_ensembles=target_ensembles,  n_pcs=argsdict["num_place_cells"],  n_hdcs=argsdict["num_headD_cells"]).to(device)
start_epoch = 0
if argsdict["use_saved_model"]:
    saved_model_file = get_latest_model_file(argsdict["save_dir"])
    model.load_state_dict(torch.load(saved_model_file))
    model.to(device)
    start_epoch = get_model_epoch(saved_model_file)
    print("RESTORING MODEL AT:", saved_model_file)
    print("STARTING AT EPOCH:", start_epoch)

# Definition of the loss
def get_loss(logits_pc, logits_hd, pc_targets, hd_targets, bottleneck_acts):
    logsoftmax = nn.LogSoftmax(dim=-1)
    pc_loss = torch.sum(-pc_targets * logsoftmax(logits_pc), -1)
    hd_loss = torch.sum(-hd_targets * logsoftmax(logits_hd), -1)
    return torch.mean(pc_loss + hd_loss)


# Definition of the Optimizer
optimiser = torch.optim.RMSprop(
    model.parameters(), lr=argsdict["lr"], momentum=argsdict["momentum"], alpha=0.9, eps=1e-10
)

if __name__ == "__main__":
    print(argsdict)
    print("USING DEVICE:", device)
    print("DEVICE PROPERTIES:", torch.cuda.get_device_properties(0))
    torch.save(target_ensembles, argsdict["save_dir"] + "target_ensembles.pt")
    torch.save(model.state_dict(), argsdict["save_dir"] + "model_epoch_0.pt")
    all_train_losses = []
    all_eval_losses = []
    times = []

    for e in tqdm(range(start_epoch, argsdict["num_epochs"])):
        t0 = time.time()
        # TRAIN MODEL
        model.train()
        step = 0
        losses = []
        for X, y in data_generator:
            print(X, y)
            break
            step += 1
        print(f"EPOCH {e}")
        print(f"training loss : {torch.mean(torch.Tensor(losses))}")
        all_train_losses.append((e, torch.mean(torch.Tensor(losses)).item()))
        times.append(time.time() - t0)
