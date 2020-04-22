# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Helper functions for creating the training graph and plotting.


--------

Adapted for pytorch, 2019

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import torch

from ensembles import PlaceCellEnsemble, HeadDirectionCellEnsemble

np.seterr(invalid="ignore")

def get_place_cell_ensembles(
    env_size, neurons_seed, n_pc, targets_type="softmax", lstm_init_type="softmax", pc_scale=[0.01], radial=False, radius=60
):
    """Create the ensembles for the Place cells."""
    if not radial:
        place_cell_ensembles = [
            PlaceCellEnsemble(
                n,
                stdev=s,
                pos_min=-env_size / 2.0,
                pos_max=env_size / 2.0,
                seed=neurons_seed,
                soft_targets=targets_type,
                soft_init=lstm_init_type,
            )
            for n, s in zip(n_pc, pc_scale)
        ]
    else:
        place_cell_ensembles = [
            PlaceCellEnsemble(
                n,
                stdev=s,
                seed=neurons_seed,
                soft_targets=targets_type,
                soft_init=lstm_init_type,
                radial=radial,
                radius=radius,
            )
            for n, s in zip(n_pc, pc_scale)
        ]
    return place_cell_ensembles


def get_head_direction_ensembles(
    neurons_seed, n_hdc, targets_type="softmax", lstm_init_type="softmax", hdc_concentration=[20.0], radial=False, radius=60
):
    """Create the ensembles for the Head direction cells."""
    if not radial:
        head_direction_ensembles = [
            HeadDirectionCellEnsemble(
                n, concentration=con, seed=neurons_seed, soft_targets=targets_type, soft_init=lstm_init_type
            )
            for n, con in zip(n_hdc, hdc_concentration)
        ]
    else:
        head_direction_ensembles = [
            HeadDirectionCellEnsemble(
                n, seed=neurons_seed, soft_targets=targets_type, soft_init=lstm_init_type, radial=radial,
            )
            for n in n_hdc
        ]
    return head_direction_ensembles


def encode_inputs(X, y, place_cell_ensembles, head_direction_ensembles, device, radial=False, coder=None):
    init_pos, init_hd, inputs = X
    target_pos, target_hd = y

    initial_conds = encode_initial_conditions(init_pos, init_hd, place_cell_ensembles, head_direction_ensembles, radial)
    ensembles_targets = encode_targets(target_pos, target_hd, place_cell_ensembles, head_direction_ensembles, radial)

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
    return (inputs, initial_conds, ensembles_targets)


def decode_outputs(outs, ensembles_targets, device, N_PC, N_HDC, coder=None):
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


def encode_initial_conditions(init_pos, init_hd, place_cell_ensembles, head_direction_ensembles, radial=False):
    initial_conds = []
    for ens in place_cell_ensembles:
        initial_conds.append(torch.squeeze(ens.get_init(init_pos[:, None, :], radial), dim=1))
    for ens in head_direction_ensembles:
        initial_conds.append(torch.squeeze(ens.get_init(init_hd[:, None, :], radial), dim=1))
    return initial_conds


def encode_targets(target_pos, target_hd, place_cell_ensembles, head_direction_ensembles, radial=False):
    ensembles_targets = []
    for ens in place_cell_ensembles:
        ensembles_targets.append(ens.get_targets(target_pos, radial))
    for ens in head_direction_ensembles:
        ensembles_targets.append(ens.get_targets(target_hd, radial))
    return ensembles_targets


def clip_all_gradients(g, var, limit):
    # print(var.name)
    return (torch.clamp(g, -limit, limit), var)


def clip_bottleneck_gradient(g, var, limit):
    if "bottleneck" in var.name or "pc_logits" in var.name:
        return (torch.clamp(g, -limit, limit), var)
    else:
        return (g, var)


def no_clipping(g, var):
    return (g, var)


def concat_dict(acc, new_data):
    """Dictionary concatenation function."""

    def to_array(kk):
        if isinstance(kk, np.ndarray):
            return kk
        else:
            return np.asarray([kk])

    for k, v in new_data.items():
        if isinstance(v, dict):
            if k in acc:
                acc[k] = concat_dict(acc[k], v)
            else:
                acc[k] = concat_dict(dict(), v)
        else:
            v = to_array(v)
            if k in acc:
                acc[k] = np.concatenate([acc[k], v])
            else:
                acc[k] = np.copy(v)
    return acc


def get_scores_and_plot(
    scorer,
    data_abs_xy,
    activations,
    directory,
    filename,
    plot_graphs=True,  # pylint: disable=unused-argument
    nbins=20,  # pylint: disable=unused-argument
    cm="jet",
    sort_by_score_60=True,
):
    """Plotting function."""

    # Concatenate all trajectories
    xy = data_abs_xy.reshape(-1, data_abs_xy.shape[-1])
    act = activations.reshape(-1, activations.shape[-1])
    n_units = act.shape[1]
    # Get the rate-map for each unit
    s = [scorer.calculate_ratemap(xy[:, 0], xy[:, 1], act[:, i]) for i in range(n_units)]
    # Get the scores
    score_60, score_90, max_60_mask, max_90_mask, sac = zip(*[scorer.get_scores(rate_map) for rate_map in s])
    # Separations
    # separations = map(np.mean, max_60_mask)
    # Sort by score if desired
    if sort_by_score_60:
        ordering = np.argsort(-np.array(score_60))
    else:
        ordering = range(n_units)
    # Plot
    cols = 16
    rows = int(np.ceil(n_units / cols))
    fig = plt.figure(figsize=(24, rows * 4))
    for i in range(n_units):
        rf = plt.subplot(rows * 2, cols, i + 1)
        acr = plt.subplot(rows * 2, cols, n_units + i + 1)
        if i < n_units:
            index = ordering[i]
            title = "%d (%.2f)" % (index, score_60[index])
            # Plot the activation maps
            scorer.plot_ratemap(s[index], ax=rf, title=title, cmap=cm)
            # Plot the autocorrelation of the activation maps
            _ = scorer.plot_sac(sac[index], mask_params=max_60_mask[index], ax=acr, title=title, cmap=cm)
    # Save
    if plot_graphs:
        if not os.path.exists(directory):
            os.makedirs(directory)
        with PdfPages(os.path.join(directory, filename), "w") as f:
            plt.savefig(f, format="pdf")
        plt.close(fig)
        print(f"Saved file: {filename}")
    return (
        np.asarray(score_60),
        np.asarray(score_90),
        np.asarray(map(np.mean, max_60_mask)),
        np.asarray(map(np.mean, max_90_mask)),
    )
