"""Code adapted from https://github.com/okrasolar/pytorch-timeseries which itself is based
on https://github.com/cauchyturing/UCR_Time_Series_Classification_Deep_Learning_Baseline/blob/master/FCN.py
"""

import torch
import torch.nn as nn


import torch.nn.functional as F


class Conv1dSamePadding(nn.Conv1d):
    def forward(self, input):
        return conv1d_same_padding(
            input, self.weight, self.bias, self.stride, self.dilation, self.groups
        )


def conv1d_same_padding(input, weight, bias, stride, dilation, groups):
    # stride and dilation are expected to be tuples.
    kernel, dilation, stride = weight.size(2), dilation[0], stride[0]
    l_out = l_in = input.size(2)
    padding = ((l_out - 1) * stride) - l_in + (dilation * (kernel - 1)) + 1
    if padding % 2 != 0:
        input = F.pad(input, [0, 1])

    return F.conv1d(
        input=input,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding // 2,
        dilation=dilation,
        groups=groups,
    )


class ConvBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, stride: int
    ) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            Conv1dSamePadding(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
            ),
            nn.BatchNorm1d(num_features=out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class FCN(nn.Module):
    def __init__(self, in_channels: int, num_classes: int = 1) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            ConvBlock(in_channels, 128, 8, 1),
            ConvBlock(128, 256, 5, 1),
            ConvBlock(256, 128, 3, 1),
        )
        self.final = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        return self.final(x.mean(dim=-1))


class LSTMFCN(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        in_channels: int,
        num_layers: int = 1,
        num_classes: int = 1,
        dropout: float = 0,
        rnn_dropout: float = 0,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()

        self.fcn = nn.Sequential(
            ConvBlock(in_channels, 128, 8, 1),
            ConvBlock(128, 256, 5, 1),
            ConvBlock(256, 128, 3, 1),
        )

        self.hidden_size = hidden_size // (1 + bidirectional)
        
        self.lstm = nn.LSTM(
            1,
            self.hidden_size,
            num_layers,
            batch_first=True,
            dropout=(num_layers > 1) * rnn_dropout,
            bidirectional=bidirectional,
        )
        
        self.final = nn.Linear(128 + self.hidden_size, num_classes)

        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_layers = num_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fcn = self.fcn(x)
        x = x.transpose(2, 1)
        emb = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        out, emb = self.lstm(emb)
        if isinstance(emb, tuple):
            emb, _ = emb
        if self.bidirectional:
            # (num_layers * num_directions x batch x hidden_size)
            emb = emb.view(self.num_layers, self.bidirectional + 1, x.size(0), -1)
            emb = emb[-1]  # take last layer
            emb = torch.cat([emb[-2], emb[-1]], 1)
        else:
            emb = out
        emb = emb.transpose(2, 1)
        x = torch.cat([x_fcn, emb], 1)
        return self.final(x.mean(dim=-1))
