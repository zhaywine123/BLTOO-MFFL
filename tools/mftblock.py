from copy import deepcopy

import torch
import torch.nn as nn


# block for mft-dual-model
class Basciblock(nn.Module):
       def __init__(
        self, 
        channels,
        channels_change,
        stride,
        padding,
        dilation,
        affine,
        cnn_layers,
        cnn_kernel_size,
        activation,
        pool_or_not,
        pooling,
       ):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # get the channels for input and output
        self.in_dim = channels
        
        if channels_change == 0:
            self.out_dim = channels//2
        elif channels_change == 1:
            self.out_dim = channels
        elif channels_change == 2:
            self.out_dim = channels * 2
        
        # choose the activation fuction
        if activation==0: self.activation=nn.Hardshrink()
        elif activation==1: self.activation=nn.ReLU(inplace=True)
        elif activation==2: self.activation=nn.LeakyReLU(inplace=True)
        elif activation==3: self.activation=nn.PReLU()
        elif activation==4: self.activation=nn.ELU()
        elif activation==5: self.activation=nn.Tanh()
        elif activation==6: self.activation=nn.Hardswish()
            
        # choose the pooling fuction
        if pooling == 0:  self.pool = "avg"
        elif pooling == 1: self.pool = "max"
            
        self.pool_op=POOLING(self.out_dim, self.out_dim, stride, self.pool, affine, True)
        
        self.layers.append(self.activation)
        
        for i in range(0, cnn_layers):
            layer = nn.Conv2d(
                self.in_dim,
                self.in_dim,
                cnn_kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=True
            )
            self.layers.append(layer)
        # last cnn layer
        layer = nn.Conv2d(
                self.in_dim,
                self.out_dim,
                cnn_kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False)
        self.layers.append(layer)
            
        self.batchnorm = nn.BatchNorm2d(self.out_dim, affine=affine, track_running_stats=True)
        
        self.layers.append(self.batchnorm)
        
        if pool_or_not == 1 : self.layers.append(self.pool_op)
        
       def forward(self, inputs):
        out = inputs
        for layer in self.layers:
            out = layer(out)
        return out


    
class POOLING(nn.Module):
    def __init__(self, C_in, C_out, stride, mode, affine=True, track_running_stats=True):
        super().__init__()
        if C_in == C_out:
            self.preprocess = None
        else:
            self.preprocess = ReLUConvBN(
                C_in, C_out, 1, 1, 0, 1, affine, track_running_stats
            )
        if mode == "avg":
            self.op = nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False)
        elif mode == "max":
            self.op = nn.MaxPool2d(3, stride=stride, padding=1)
        else:
            raise ValueError(f"Invalid mode={mode} in POOLING")

    def forward(self, inputs):
        if self.preprocess:
            x = self.preprocess(inputs)
        else:
            x = inputs
        return self.op(x)
