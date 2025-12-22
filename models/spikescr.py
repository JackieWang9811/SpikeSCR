from functools import partial
import torch
import torch.nn as nn

from torch.nn.init import xavier_uniform_
from spikingjelly.activation_based import neuron, layer
import numpy as np
from spikingjelly.activation_based.neuron import LIFNode,ParametricLIFNode
from module.conv import Transpose
from spikingjelly.datasets import padded_sequence_mask
from module.spikscr_module import MS_Block_Conv


import math
import time


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model) # (5000,840)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # (5000,1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) #  (420,) # 840/2
        pe[:, 0::2] = torch.sin(position * div_term) #
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) # (5000,1,840)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class SpikingEmbed(nn.Module):
    def __init__(self, config):
        super(SpikingEmbed, self).__init__()
        self.config = config

        self.conv = nn.Conv1d(config.n_inputs, config.n_hidden_neurons, kernel_size=3, stride=1, bias=config.bias, padding=1)
        if self.config.use_bn:
            self.bn = layer.BatchNorm1d(config.n_hidden_neurons, step_mode='m')
        if self.config.use_dp:
            self.dropout = layer.Dropout(config.dropout_l, step_mode='m')

        self.trans1 = Transpose(0,2,1)
        self.trans2 = Transpose(2,0,1)

        if self.config.spike_mode == 'lif':
            self.lif = neuron.LIFNode(
                tau=config.init_tau,
                v_threshold=config.v_threshold,
                surrogate_function=config.surrogate_function,
                detach_reset=config.detach_reset,
                step_mode='m',
                decay_input=False,
                store_v_seq=True,
                backend=config.backend
            )
        elif self.config.spike_mode == 'plif':
            self.lif = ParametricLIFNode(
                init_tau=config.init_tau,
                v_threshold=config.v_threshold,
                surrogate_function=config.surrogate_function,
                detach_reset=config.detach_reset,
                step_mode='m',
                decay_input=False,
                store_v_seq=True,
                backend=config.backend
            )

    def forward(self, x):
        x = self.trans1(x)
        x = self.conv(x)
        x = self.trans2(x)
        if self.config.use_bn:
            x = self.bn(x)
        x = self.lif(x)
        if self.config.use_dp:
            x = self.dropout(x)
        return x


class SpikeDrivenTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.spike_embed = SpikingEmbed(config)

        self.blocks = nn.ModuleList(
            [
                MS_Block_Conv(
                    config=self.config,
                    dim=self.config.n_hidden_neurons,
                    num_heads=self.config.num_heads,
                    init_tau = self.config.init_tau,
                    spike_mode=self.config.spike_mode,
                    layers=j,
                )
                for j in range(self.config.depths)
            ]
        )

        if self.config.use_norm:
            self.final_dp = layer.Dropout(self.config.dropout_l, step_mode='m')
            if self.config.spike_mode == 'lif':
                self.final_lif = LIFNode(tau=self.config.init_tau, v_threshold=self.config.v_threshold,
                                        surrogate_function=self.config.surrogate_function,
                                        detach_reset=self.config.detach_reset,
                                        step_mode='m', decay_input=False, store_v_seq=True, backend=config.backend)
            elif self.config.spike_mode == 'plif':
                self.final_lif = ParametricLIFNode(init_tau=config.init_tau, v_threshold=config.v_threshold,
                                      surrogate_function=config.surrogate_function,
                                      detach_reset=config.detach_reset,
                                      step_mode='m', decay_input=False, store_v_seq=True, backend=config.backend)
            self.head = layer.Linear(self.config.n_hidden_neurons, self.config.n_outputs, bias=False,step_mode='m')

        else:
            if self.config.use_dp:
                self.final_dp = layer.Dropout(self.config.dropout_l, step_mode='m')

            self.head = layer.Linear(self.config.n_hidden_neurons, self.config.n_outputs, bias=False,step_mode='m')

        self._reset_parameters()


    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, x, attention_mask):

        x = self.spike_embed(x)

        for module in self.blocks:
            x = module(x, attention_mask)


        if self.config.use_norm:
            x = self.head(self.final_dp(self.final_lif(x)))
        else:
            if self.config.use_dp:
                x = self.head(self.final_dp(x))
            else:
                x = self.head(x)

        return x


