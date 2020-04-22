import torch
from torch import nn
import numpy as np


def truncated_normal_(tensor, mean=0, std=1):
    # https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/15
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor


def init_trunc_normal(t, size):
    std = 1.0 / np.sqrt(size)
    return truncated_normal_(t, 0, std)


def load_tf_param(loc):
    return nn.Parameter(torch.Tensor(np.load(loc).T))


class GridTorch(nn.Module):
    def __init__(
        self,
        target_ensembles,  # Targets, place cells and head direction cells
        nh_lstm=128,  # Size of LSTM cell
        nh_bottleneck=256,  # Size of the linear layer between LSTM output and output
        n_pcs=256,  # number of place cells
        n_hdcs=12,  # number of head direction cells
        dropoutrates_bottleneck=0.5,  # Dropout rate at bottleneck
        bottleneck_has_bias=False,
        disable_LSTM_training=False, # Use pretrained LSTM
    ):
        super().__init__()
        self.target_ensembles = target_ensembles
        self.disable_LSTM_training = disable_LSTM_training
        # Weights to compute the initial cell and hidden state of the LSTM
        self.state_embed = nn.Linear(n_pcs + n_hdcs, nh_lstm)  # weight W^cp and W^cd
        self.cell_embed = nn.Linear(n_pcs + n_hdcs, nh_lstm)  # weight W^hp and W^hd
        # Recurrent layer
        self.lstm = nn.LSTM(input_size=3, hidden_size=nh_lstm)

        self.bottleneck = nn.Linear(nh_lstm, nh_bottleneck, bias=bottleneck_has_bias)
        self.non_linearity = nn.Tanh()
        self.pc_logits = nn.Linear(nh_bottleneck, target_ensembles[0].n_cells)
        self.hd_logits = nn.Linear(nh_bottleneck, target_ensembles[1].n_cells)

        self.dropout = nn.Dropout(dropoutrates_bottleneck)

        with torch.no_grad():
            self.state_embed.weight = init_trunc_normal(self.state_embed.weight, 128)
            self.cell_embed.weight = init_trunc_normal(self.cell_embed.weight, 128)
            self.bottleneck.weight = init_trunc_normal(self.bottleneck.weight, 256)
            self.pc_logits.weight = init_trunc_normal(self.pc_logits.weight, 256)
            self.hd_logits.weight = init_trunc_normal(self.hd_logits.weight, 12)
            nn.init.kaiming_uniform_(self.lstm.weight_ih_l0)
            nn.init.kaiming_uniform_(self.lstm.weight_hh_l0)
            nn.init.zeros_(self.state_embed.bias)
            nn.init.zeros_(self.cell_embed.bias)
            nn.init.zeros_(self.pc_logits.bias)
            nn.init.zeros_(self.hd_logits.bias)
            nn.init.zeros_(self.lstm.bias_hh_l0)
            nn.init.zeros_(self.lstm.bias_ih_l0)

    @property
    def l2_loss(self,):
        return self.bottleneck.weight.norm(2) + self.pc_logits.weight.norm(2) + self.hd_logits.weight.norm(2)

    def forward(self, x, initial_conds):
        batch_size = x.shape[1]
        init = torch.cat(initial_conds, dim=1)  # Shape: batch_size x 268
        init_state = self.state_embed(init)  # l_0; Shape: batch_size x 128
        init_cell = self.cell_embed(init)  # m_0; Shape: batch_size x 128
        h_t, c_t = init_state, init_cell

        logits_hd = []
        logits_pc = []
        bottleneck_acts = []
        rnn_states = []
        cell_states = []
        for t in x:
            # get lstm output predictions
            _, (h_t, c_t) = self.lstm(t.view(1, batch_size, -1), (h_t.unsqueeze(0), c_t.unsqueeze(0)))
            h_t, c_t = h_t.squeeze(), c_t.squeeze()
            if self.disable_LSTM_training:
                h_t, c_t = h_t.detach(), c_t.detach()
            # The ratemaps take the weights of the battleneck activation (without the dropout)
            non_linearity = self.non_linearity(self.bottleneck(h_t))
            bottleneck_activations = self.dropout(non_linearity)

            pc_preds = self.pc_logits(bottleneck_activations)
            hd_preds = self.hd_logits(bottleneck_activations)

            logits_hd += [hd_preds]
            logits_pc += [pc_preds]
            bottleneck_acts += [bottleneck_activations]
            rnn_states += [h_t]
            cell_states += [c_t]

        outs = (
            torch.stack(logits_hd),
            torch.stack(logits_pc),
            torch.stack(bottleneck_acts),
            torch.stack(rnn_states),
            torch.stack(cell_states),
        )
        return outs
