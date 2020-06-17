"""Code adapted from https://github.com/okrasolar/pytorch-timeseries which itself is based
on https://github.com/cauchyturing/UCR_Time_Series_Classification_Deep_Learning_Baseline/blob/master/FCN.py
"""

import torch
import torch.nn as nn

import torch.nn.functional as F

from utils import get_arguments


def make_mlp_layer(dropout, input_size, output_size):
    mappings = [nn.ReLU(inplace=True)]
    if dropout > 0:
        mappings.append(nn.Dropout(dropout))
    mappings.append(nn.Linear(input_size, output_size))
    return nn.Sequential(*mappings)


class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_size, layers=(128, 128, 128), dropout=0.0):
        super().__init__()

        mappings = [nn.Linear(input_size, layers[0])]
        for i in range(1, len(layers)):
            mappings.append(make_mlp_layer(dropout, layers[i - 1], layers[i]))

        mappings.append(nn.ReLU(inplace=True))
        mappings.append(nn.Linear(layers[-1], 1))
        self.mappings = nn.Sequential(*mappings)

    def forward(self, x):
        return self.mappings(x)


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        return x.mean(dim=-1) # GAP


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        x = self.layers(x)
        return x.mean(dim=-1)


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


def identity(x):
    return x

def shortcut(c_in, c_out):
    return torch.nn.Sequential(*[torch.nn.Conv1d(c_in, c_out, kernel_size=1), 
                                 torch.nn.BatchNorm1d(c_out)])
    
class Inception(torch.nn.Module):
    def __init__(self, c_in, bottleneck=32, ks=40, nb_filters=32):

        super().__init__()
        self.bottleneck = torch.nn.Conv1d(c_in, bottleneck, 1) if bottleneck and c_in > 1 else identity
        mts_feat = bottleneck or c_in
        conv_layers = []
        kss = [ks // (2**i) for i in range(3)]
        # ensure odd kss until torch.nn.Conv1d with padding='same' is available in pytorch 1.3
        kss = [ksi if ksi % 2 != 0 else ksi - 1 for ksi in kss]  
        for i in range(len(kss)):
            conv_layers.append(
                torch.nn.Conv1d(mts_feat, nb_filters, kernel_size=kss[i], padding=kss[i] // 2))
        self.conv_layers = torch.nn.ModuleList(conv_layers)
        self.maxpool = torch.nn.MaxPool1d(3, stride=1, padding=1)
        self.conv = torch.nn.Conv1d(c_in, nb_filters, kernel_size=1)
        self.bn = torch.nn.BatchNorm1d(nb_filters * 4)
        self.act = torch.nn.ReLU()

    def forward(self, x):
        input_tensor = x
        x = self.bottleneck(input_tensor)
        for i in range(3):
            out_ = self.conv_layers[i](x)
            if i == 0:
                out = out_
            else:
                out = torch.cat((out, out_), 1)
        mp = self.conv(self.maxpool(input_tensor))
        inc_out = torch.cat((out, mp), 1)
        return self.act(self.bn(inc_out))


class InceptionBlock(torch.nn.Module):
    def __init__(self, c_in, bottleneck=32, ks=40, nb_filters=32, residual=True, depth=6):

        super().__init__()

        self.residual = residual
        self.depth = depth

        #inception & residual layers
        inc_mods = []
        res_layers = []
        res = 0
        for d in range(depth):
            inc_mods.append(
                Inception(c_in if d == 0 else nb_filters * 4,
                          bottleneck=bottleneck if d > 0 else 0,
                          ks=ks,
                          nb_filters=nb_filters))
            if self.residual and d % 3 == 2:
                res_layers.append(shortcut(c_in if res == 0 else nb_filters * 4, nb_filters * 4))
                res += 1
            else:
                res_layer = res_layers.append(None)
        self.inc_mods = torch.nn.ModuleList(inc_mods)
        self.res_layers = torch.nn.ModuleList(res_layers)
        self.act = torch.nn.ReLU()
        
    def forward(self, x):
        res = x
        for d, l in enumerate(range(self.depth)):
            x = self.inc_mods[d](x)
            if self.residual and d % 3 == 2:
                res = self.res_layers[d](res)
                x += res
                res = x
                x = self.act(x)
        return x
    
class InceptionTime(torch.nn.Module):
    def __init__(self,c_in, c_out, bottleneck=32, ks=40, nb_filters=32, residual=True, depth=6):
        super().__init__()

        self.input_args = get_arguments()
        
        self.block = InceptionBlock(c_in, bottleneck=bottleneck, ks=ks, nb_filters=nb_filters,
                                    residual=residual,depth=depth)
        self.gap = torch.nn.AdaptiveAvgPool1d(1)
        self.fc = torch.nn.Linear(nb_filters * 4, c_out)

    def forward(self, x):
        x = self.block(x)
        x = self.gap(x).squeeze(-1)
        x = self.fc(x)
        return x
