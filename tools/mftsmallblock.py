from copy import deepcopy

import torch
import torch.nn as nn
BLOCKCHOSE = {
    "basicblock_1x1": lambda channels, channels_change, activation, pooling_or_not, pooling_mode, affine, track_running_stats: Basciblock(
        channels,channels_change,
        activation, pooling_or_not, pooling_mode,
        (1, 1), (1, 1), (0, 0), (1, 1),
        affine = True,
        track_running_stats = True
    ),
    "basicblock_3x3": lambda channels, channels_change, activation, pooling_or_not, pooling_mode, affine, track_running_stats: Basciblock(
        channels,channels_change,
        activation, pooling_or_not, pooling_mode,
        (3, 3), (1, 1), (1, 1), (1, 1),
        affine = True,
        track_running_stats = True
    ),
    "basicblock_5x5": lambda channels, channels_change, activation, pooling_or_not, pooling_mode, affine, track_running_stats: Basciblock(
        channels,channels_change,
        activation, pooling_or_not, pooling_mode,
        (5, 5), (1, 1), (2, 2), (1, 1),
        affine = True,
        track_running_stats = True
    ),
    "basicblock_7x7": lambda channels, channels_change, activation, pooling_or_not, pooling_mode, affine, track_running_stats: Basciblock(
        channels,channels_change,
        activation, pooling_or_not, pooling_mode,
        (7, 7), (1, 1), (3, 3), (1, 1),
        affine = True,
        track_running_stats = True
    ),
    "sepblock_3x3": lambda channels, channels_change, activation, pooling_or_not, pooling_mode, affine, track_running_stats: Sepblock(
        channels,channels_change,
        activation, pooling_or_not, pooling_mode,
        (3, 3), (1, 1), (1, 1), (1, 1),
        affine = True,
        track_running_stats = True
    ),
    "sepblock_5x5": lambda channels, channels_change, activation, pooling_or_not, pooling_mode, affine, track_running_stats: Sepblock(
        channels,channels_change,
        activation, pooling_or_not, pooling_mode,
        (5, 5), (1, 1), (2, 2), (1, 1),
        affine = True,
        track_running_stats = True
    ),
    "sepblock_7x7": lambda channels, channels_change, activation, pooling_or_not, pooling_mode, affine, track_running_stats: Sepblock(
        channels,channels_change,
        activation, pooling_or_not, pooling_mode,
        (7, 7), (1, 1), (3, 3), (1, 1),
        affine = True,
        track_running_stats = True
    ),
    "dilblock_3x3": lambda channels, channels_change, activation, pooling_or_not, pooling_mode, affine, track_running_stats: Sepblock(
        channels,channels_change,
        activation, pooling_or_not, pooling_mode,
        (3, 3), (1, 1), (2, 2), (2, 2),
        affine = True,
        track_running_stats = True
    ),
    "dilblock_5x5": lambda channels, channels_change, activation, pooling_or_not, pooling_mode, affine, track_running_stats: Sepblock(
        channels,channels_change,
        activation, pooling_or_not, pooling_mode,
        (5, 5), (1, 1), (4, 4), (2, 2),
        affine = True,
        track_running_stats = True
    ),
    "dilblock_7x7": lambda channels, channels_change, activation, pooling_or_not, pooling_mode, affine, track_running_stats: Sepblock(
        channels,channels_change,
        activation, pooling_or_not, pooling_mode,
        (7, 7), (1, 1), (6, 6), (2, 2),
        affine = True,
        track_running_stats = True
    ),
    "dilsepblock_3x3": lambda channels, channels_change, activation, pooling_or_not, pooling_mode, affine, track_running_stats: DilSepblock(
        channels,channels_change,
        activation, pooling_or_not, pooling_mode,
        3, 1, 1, 1,
        affine = True,
        track_running_stats = True
    ),
    "dilsepblock_5x5": lambda channels, channels_change, activation, pooling_or_not, pooling_mode, affine, track_running_stats: DilSepblock(
        channels,channels_change,
        activation, pooling_or_not, pooling_mode,
        3, 1, 2, 1,
        affine = True,
        track_running_stats = True
    ),
    "dilsepblock_7x7": lambda channels, channels_change, activation, pooling_or_not, pooling_mode, affine, track_running_stats: DilSepblock(
        channels,channels_change,
        activation, pooling_or_not, pooling_mode,
        3, 1, 3, 1,
        affine = True,
        track_running_stats = True
    ),
    "skip_connect": lambda channels, channels_change, activation, pooling_or_not, pooling_mode, affine, track_running_stats: Identity(channels)
}

# small block for mft-dual-model
class Basciblock(nn.Module):
    def __init__(
        self, 
        channels,
        channels_change,
        activation,
        pooling_or_not,
        pooling_mode,
        kernel_size,
        stride,
        padding,
        dilation,
        affine,
        track_running_stats=True
       ):
        super().__init__()
        
        self.pooling_or_not = pooling_or_not
        # get the channels for input and output
        
        self.in_dim = channels
        if channels_change == 0:
            self.out_dim = channels//2
        elif channels_change == 1:
            self.out_dim = channels
        elif channels_change == 2:
            self.out_dim = channels * 2
        
        if self.out_dim < 8 : self.out_dim = 8
        if self.out_dim > 256 : self.out_dim = 256
            
        # choose the activation fuction
        if activation==0: self.activation=nn.Hardshrink()
        elif activation==1: self.activation=nn.ReLU()
        elif activation==2: self.activation=nn.LeakyReLU()
        elif activation==3: self.activation=nn.PReLU()
        elif activation==4: self.activation=nn.ELU()
        elif activation==5: self.activation=nn.Tanh()
        elif activation==6: self.activation=nn.Hardswish()    
        
        
        self.op = nn.Sequential(
            self.activation,
            nn.Conv2d(
                self.in_dim,
                self.out_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=not affine,
            ),
            nn.BatchNorm2d(self.out_dim, affine=affine, track_running_stats=track_running_stats),
        )
        

        if pooling_mode == "avg":
            self.poolop = nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False)
        elif pooling_mode == "max":
            self.poolop = nn.MaxPool2d(3, stride=stride, padding=1)
        elif pooling_mode == "none":
            self.poolop = None
            
    def forward(self, inputs):
        out = self.op(inputs)
        if self.pooling_or_not == True: 
            out = self.poolop(out)
        return out

class DownSampleblock(nn.Module):
    def __init__(
        self, 
        in_dim,
        out_dim,
        activation,
        kernel_size = 1,
        stride = 1,
        padding = 0,
        dilation = 0,
        affine = True,
        track_running_stats=True
       ):
        super().__init__()
        # get the channels for input and output
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # choose the activation fuction
        if activation==0: self.activation=nn.Hardshrink()
        elif activation==1: self.activation=nn.ReLU()
        elif activation==2: self.activation=nn.LeakyReLU()
        elif activation==3: self.activation=nn.PReLU()
        elif activation==4: self.activation=nn.ELU()
        elif activation==5: self.activation=nn.Tanh()
        elif activation==6: self.activation=nn.Hardswish()    
        
        
        self.op = nn.Sequential(
            self.activation,
            nn.Conv2d(
                self.in_dim,
                self.out_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=not affine,
            ),
            nn.BatchNorm2d(self.out_dim, affine=affine, track_running_stats=track_running_stats),
        )
        
            
    def forward(self, inputs):
        out = self.op(inputs)
        return out
    
    
class Sepblock(nn.Module):
    def __init__(
        self, 
        channels,
        channels_change,
        activation,
        pooling_or_not,
        pooling_mode,
        kernel_size,
        stride,
        padding,
        dilation,
        affine,
        track_running_stats=True
       ):
        super().__init__()
        
        self.pooling_or_not = pooling_or_not
        # get the channels for input and output
        
        self.in_dim = channels
        if channels_change == 0:
            self.out_dim = channels//2
        elif channels_change == 1:
            self.out_dim = channels
        elif channels_change == 2:
            self.out_dim = channels * 2
        if self.out_dim < 8 : self.out_dim = 8
        if self.out_dim > 256 : self.out_dim = 256        
        # choose the activation fuction
        if activation==0: self.activation=nn.Hardshrink()
        elif activation==1: self.activation=nn.ReLU()
        elif activation==2: self.activation=nn.LeakyReLU()
        elif activation==3: self.activation=nn.PReLU()
        elif activation==4: self.activation=nn.ELU()
        elif activation==5: self.activation=nn.Tanh()
        elif activation==6: self.activation=nn.Hardswish()    
        
        
        self.op = nn.Sequential(
            self.activation,
            nn.Conv2d(
                self.in_dim,
                self.in_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=not affine,
            ),
            nn.Conv2d(self.in_dim, self.out_dim, kernel_size=1, padding=0, bias=not affine),
            nn.BatchNorm2d(self.out_dim, affine=affine, track_running_stats=track_running_stats),
        )
        

        if pooling_mode == "avg":
            self.poolop = nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False)
        elif pooling_mode == "max":
            self.poolop = nn.MaxPool2d(3, stride=stride, padding=1)
        elif pooling_mode == "none":
            self.poolop = None
            
    def forward(self, inputs):
        out = self.op(inputs)
        if self.pooling_or_not == True: 
            out = self.poolop(out)
        return out       
    
class DilSepblock(nn.Module):
    def __init__(
        self, 
        channels,
        channels_change,
        activation,
        pooling_or_not,
        pooling_mode,
        kernel_size,
        stride,
        padding,
        dilation,
        affine,
        track_running_stats=True
       ):
        super().__init__()
        
        self.pooling_or_not = pooling_or_not
        # get the channels for input and output
        
        self.in_dim = channels
        self.out_dim = channels  


        # choose the activation fuction
        if activation==0: self.activation=nn.Hardshrink()
        elif activation==1: self.activation=nn.ReLU()
        elif activation==2: self.activation=nn.LeakyReLU()
        elif activation==3: self.activation=nn.PReLU()
        elif activation==4: self.activation=nn.ELU()
        elif activation==5: self.activation=nn.Tanh()
        elif activation==6: self.activation=nn.Hardswish()    
        
        
        self.op = nn.Sequential(
            self.activation,
            nn.Conv2d(
                self.in_dim,
                self.in_dim,
                kernel_size=(1,kernel_size),
                stride=(1,stride),
                padding=(0,1),
                bias=not affine,
            ),
            nn.Conv2d(
                self.in_dim, 
                self.in_dim, 
                kernel_size=(kernel_size,1), 
                stride=(stride,1),
                padding=(1,0), 
                bias=not affine),
            nn.BatchNorm2d(self.in_dim, affine=affine, track_running_stats=track_running_stats),
        )
        

        if pooling_mode == "avg":
            self.poolop = nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False)
        elif pooling_mode == "max":
            self.poolop = nn.MaxPool2d(3, stride=stride, padding=1)
        elif pooling_mode == "none":
            self.poolop = None
            
    def forward(self, inputs):
        out = self.op(inputs)
        if self.pooling_or_not == True: 
            out = self.poolop(out)
        return out   
    
class Identity(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.out_dim = channels

    def forward(self, x):  # pylint: disable=no-self-use
        return x
    