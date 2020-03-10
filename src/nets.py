"""Code adapted from https://github.com/okrasolar/pytorch-timeseries which itself is based
on https://github.com/cauchyturing/UCR_Time_Series_Classification_Deep_Learning_Baseline/blob/master/FCN.py
"""

import torch
import torch.nn as nn

import torch.nn.functional as F

from utils import get_arguments


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

        self.input_args = get_arguments()

        self.layers = nn.Sequential(
            ConvBlock(in_channels, 128, 7, 1),
            ConvBlock(128, 256, 5, 1),
            ConvBlock(256, 128, 3, 1),
        )
        self.final = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        return self.final(x.mean(dim=-1)) # GAP


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

        self.input_args = get_arguments()

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
        
        self.final = nn.Linear(128 + self.hidden_size * (1 + bidirectional), num_classes)

        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_layers = num_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fcn = self.fcn(x)
        x = x.transpose(2, 1)
        emb = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        out, (hn, _) = self.lstm(emb)
        return self.final(torch.cat([x_fcn.mean(dim=-1), hn[-1]], 1)) # GAP


class ResNet(nn.Module):

    def __init__(self, in_channels: int, mid_channels: int = 64,
                 num_pred_classes: int = 1) -> None:
        super().__init__()

        self.input_args = get_arguments()

        # for easier saving and loading
        self.input_args = {
            'in_channels': in_channels,
            "mid_channels": mid_channels,
            'num_pred_classes': num_pred_classes
        }

        self.layers = nn.Sequential(*[
            ResNetBlock(in_channels=in_channels, out_channels=mid_channels),
            ResNetBlock(in_channels=mid_channels, out_channels=mid_channels * 2),
            ResNetBlock(in_channels=mid_channels * 2, out_channels=mid_channels * 2),

        ])
        self.final = nn.Linear(mid_channels * 2, num_pred_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        x = self.layers(x)
        return self.final(x.mean(dim=-1)) # GAP


class ResNetBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        channels = [in_channels, out_channels, out_channels, out_channels]
        kernel_sizes = [8, 5, 3]

        self.layers = nn.Sequential(*[
            ConvBlock(in_channels=channels[i], out_channels=channels[i + 1],
                      kernel_size=kernel_sizes[i], stride=1) for i in range(len(kernel_sizes))
        ])

        self.match_channels = False
        if in_channels != out_channels:
            self.match_channels = True
            self.residual = nn.Sequential(*[
                Conv1dSamePadding(in_channels=in_channels, out_channels=out_channels,
                                  kernel_size=1, stride=1),
                nn.BatchNorm1d(num_features=out_channels)
            ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore

        if self.match_channels:
            return self.layers(x) + self.residual(x)
        return self.layers(x)
