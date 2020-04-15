import torch
from torch.utils import data
from torch import nn
import numpy as np
from tqdm import tqdm
import time
import argparse
from dataloading import Dataset
from model_utils import get_latest_model_file, get_model_epoch
from model_lstm import GridTorch
import utils


from misc.rate_coding import PadCoder

# ENV_SIZE = 2.2
# N_EPOCHS = 10  # 1000
# STEPS_PER_EPOCH = 100
# BATCH_SIZE = 128
# GRAD_CLIPPING = 1e-5

# N_PC = [256]
# N_HDC = [12]
# BOTTLENECK_DROPOUT = 0.5
# WEIGHT_DECAY = 1e-5
# LR = 1e-5
# MOMENTUM = 0.9
# SAVE_LOC = "./experiments/"
# USE_PREVIOUSLY_TRAINED_MODEL = False
#
# SEED = 9101
# torch.manual_seed(SEED)
# np.random.seed(SEED)

parser = argparse.ArgumentParser(description='PyTorch Grid Cells Path Integration')

parser.add_argument('--env_size', type=int, default=2.2, help='size of environment')
parser.add_argument('--num_epochs', type=int, default=10, help='number of epochs to stop after')
parser.add_argument('--steps', type=int, default=100, help='steps per epoch')
parser.add_argument('--batch_size', type=int, default=128, help='size of one minibatch')
parser.add_argument('--grad_clip', type=float, default=1e-5, help='gradient clipping')

parser.add_argument('--num_place_cells', type=int, default=256, help='number of epochs to stop after')
parser.add_argument('--num_headD_cells', type=int, default=12, help='number of epochs to stop after')
parser.add_argument('--btln_dropout', type=float, default=1e-5, help='bottleneck dropout')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay')
parser.add_argument('--lr', type=float, default=1e-5, help='initial learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

parser.add_argument('--hidden_size', type=int, default=256, help='size of hidden layers')

parser.add_argument('--save_dir', type=str, default='./experiments/results/',
                    help='path to save the experimental config, logs, model \
                    This is not automatically generated.')

parser.add_argument('--use_saved_model', type=bool, default=False, help='Use previously trained model')
parser.add_argument('--seed', type=int, default=9999, help='random seed')


args = parser.parse_args()
argsdict = args.__dict__

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
print("USING DEVICE:", device)

# Parameters
data_params = {
    "batch_size":  argsdict['batch_size'],
    "shuffle": True,
    # "num_workers": 1,  # num cpus,
    "num_workers": 6,  # num cpus,
}

test_params = {
    "batch_size": 100,
    "shuffle": True,
    "num_workers": 2,  # num cpus,
}

dataset = Dataset(batch_size=data_params["batch_size"])
data_generator = data.DataLoader(dataset, **data_params)
test_generator = data.DataLoader(dataset, **test_params)
N_PC = [argsdict['num_place_cells']]
N_HDC = [argsdict['num_headD_cells']]

# Create the ensembles that provide targets during training
place_cell_ensembles = utils.get_place_cell_ensembles(env_size=argsdict['env_size'], neurons_seed=argsdict['seed'], n_pc=N_PC)
head_direction_ensembles = utils.get_head_direction_ensembles(neurons_seed=argsdict['seed'], n_hdc=N_HDC)
target_ensembles = place_cell_ensembles + head_direction_ensembles

model = GridTorch(target_ensembles).to(device)

start_epoch = 0

if argsdict['use_saved_model']:
    saved_model_file = get_latest_model_file(argsdict['save_dir'])
    model.load_state_dict(torch.load(saved_model_file))
    model.to(device)
    start_epoch = get_model_epoch(saved_model_file)
    print("RESTORING MODEL AT:", saved_model_file)
    print("STARTING AT EPOCH:", start_epoch)

# loss ops:
logsoftmax = nn.LogSoftmax(dim=-1)


def cross_entropy(pred, soft_targets):
    return torch.sum(-soft_targets * logsoftmax(pred), -1)


# Optimisation opts
optimiser = torch.optim.RMSprop(model.parameters(), lr=argsdict['lr'], momentum=argsdict['momentum'], alpha=0.9, eps=1e-10)


def encode_inputs(X, y, place_cell_ensembles, head_direction_ensembles, coder=None):
    init_pos, init_hd, inputs = X
    target_pos, target_hd = y

    initial_conds = utils.encode_initial_conditions(init_pos, init_hd, place_cell_ensembles, head_direction_ensembles)
    ensembles_targets = utils.encode_targets(target_pos, target_hd, place_cell_ensembles, head_direction_ensembles)

    init_pos = init_pos.to(device)
    init_hd = init_hd.to(device)
    inputs = inputs.to(device)
    target_pos = target_pos.to(device)
    target_hd = target_hd.to(device)
    initial_conds = tuple(map(lambda x: x.to(device), initial_conds))

    if coder:
        inputs = coder(inputs, value=torch.Tensor([0.0, 1.0, 0.0]))
        target_pos = coder(target_pos, target=True)
        target_hd = coder(target_hd, target=True)

    inputs = inputs.transpose(1, 0)
    return (
        init_pos,
        init_hd,
        inputs,
        target_pos,
        target_hd,
        initial_conds,
        ensembles_targets,
    )


def decode_outputs(outs, ensembles_targets, coder=None):
    pc_targets = ensembles_targets[0].to(device)
    hd_targets = ensembles_targets[1].to(device)

    logits_hd, logits_pc, bottleneck_acts, lstm_states, _ = outs
    pc_targets, hd_targets = (pc_targets.transpose(1, 0), hd_targets.transpose(1, 0))

    logits_pc = logits_pc.view(-1, N_PC[0])
    logits_hd = logits_hd.view(-1, N_HDC[0])

    if coder:
        pc_targets, hd_targets = (
            coder(pc_targets, target=True),
            coder(hd_targets, target=True),
        )

    pc_targets = pc_targets.contiguous().view(-1, N_PC[0])
    hd_targets = hd_targets.contiguous().view(-1, N_HDC[0])

    return bottleneck_acts, logits_pc, logits_hd, pc_targets, hd_targets


def get_loss(logits_pc, logits_hd, pc_targets, hd_targets, bottleneck_acts):
    pc_loss = cross_entropy(logits_pc, pc_targets)
    hd_loss = cross_entropy(logits_hd, hd_targets)
    return torch.mean(pc_loss + hd_loss)


# coder = PadCoder(3)
coder = None

if __name__ == "__main__":
    torch.save(target_ensembles, argsdict['save_dir'] + "target_ensembles.pt")
    torch.save(model.state_dict(), argsdict['save_dir'] + "model_epoch_0.pt")

    for e in tqdm(range(start_epoch, argsdict['num_epochs'])):

        model.train()
        step = 0
        losses = []
        for X, y in data_generator:
            optimiser.zero_grad()
            (init_pos, init_hd, inputs, target_pos, target_hd, initial_conds, ensembles_targets,) = encode_inputs(
                X, y, place_cell_ensembles, head_direction_ensembles, coder=coder
            )
            outs = model.forward(inputs, initial_conds)
            (bottleneck_acts, logits_pc, logits_hd, pc_targets, hd_targets,) = decode_outputs(
                outs, ensembles_targets, coder=coder
            )
            loss = get_loss(logits_pc, logits_hd, pc_targets, hd_targets, bottleneck_acts)
            loss += model.l2_loss * argsdict['weight_decay']
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), argsdict['grad_clip'])
            optimiser.step()
            losses.append(loss.clone().item())
            if step > argsdict['steps']:
                break
            step += 1
        print(f"EPOCH {e} LOSS : {torch.mean(torch.Tensor(losses))}")
        # evaluation routine
        if (e+1)%2==0:
            state_dict = model.state_dict()
            for k, v in state_dict.items():
                state_dict[k] = v.cpu()
            torch.save(state_dict, argsdict['save_dir'] + "model_epoch_{}.pt".format(e))
            with torch.no_grad():
                model.eval()
                for data in test_generator:
                    test_X, test_y = data

                    (
                        init_pos,
                        init_hd,
                        inputs,
                        target_pos,
                        target_hd,
                        initial_conds,
                        ensembles_targets,
                    ) = encode_inputs(test_X, test_y, place_cell_ensembles, head_direction_ensembles, coder=coder,)

                    outs = model.forward(inputs, initial_conds)

                    (bottleneck_acts, logits_pc, logits_hd, pc_targets, hd_targets,) = decode_outputs(
                        outs, ensembles_targets, coder=coder
                    )

                    loss = get_loss(logits_pc, logits_hd, pc_targets, hd_targets, bottleneck_acts)

                    print("LOSS:", loss)

                    break
