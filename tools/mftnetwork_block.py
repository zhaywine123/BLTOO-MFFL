import torch.nn as nn


from .mftsmallblock import *
from .mftoperation import ResNetBasicblock


        
# Encoding examples
#param 'channels, channels_change, activation, pooling_or_not, pooling_mode, affine, track_running_stats'

# block arch=[block index, channel change, activation, pooling or not, poolmode]
# block encoding = [block index , channel change, activation, pooling]

#model arch = [block1,block2, ... , blockn]


# blocks 24     3-6   18-21    3 5      7   5                         5
#arch encoding = [num_Sblock, num_Dblocks, resnet_or_not_mulit,
#                 block encoding
#                 block encoding ...
#
#                 block encoding ...
#                 lr, mfl *6]

class MftNetwork(nn.Module):
    def __init__(self, C, num_blocks,num_Sblock, num_Dblocks, resnet_or_not_mulit, blocksgenotype, num_classes):
        super().__init__()
        self._C = C
        #self._layerN = N
        #block解码
        
        self.num_blocks=num_blocks
        self.num_Sblock=num_Sblock
        self.num_Dblocks=num_Dblocks
        self.resnet_or_not_mulit = resnet_or_not_mulit
        self.blocks = nn.ModuleList()
        self.downsample_op = nn.ModuleList()
        self.downsample_count=[]
        self.stem = nn.Sequential(
            nn.Conv2d(3, C, kernel_size=7, stride=2, padding=3, bias=False), nn.BatchNorm2d(C)
        )
        
        #self.out_channels=[]
        #初始Channels
        channels = C
        #out_channels.append(C)
        count=0

        for blockgenotype in blocksgenotype:
            #block encoding = [block index , channel change, activation, pooling]
            index = blockgenotype[0]
            if index == 0 : opname = "basicblock_1x1"
            elif index == 1 : opname = "basicblock_3x3"
            elif index == 2 : opname = "basicblock_5x5"
            elif index == 3 : opname = "basicblock_7x7"    
            elif index == 4 : opname = "sepblock_3x3"    
            elif index == 5 : opname = "sepblock_5x5"    
            elif index == 6 : opname = "sepblock_7x7"    
            elif index == 7 : opname = "dilblock_3x3"    
            elif index == 8 : opname = "dilblock_5x5"    
            elif index == 9 : opname = "dilblock_7x7"
            elif index == 10 : opname = "dilsepblock_3x3"    
            elif index == 11 : opname = "dilsepblock_5x5"    
            elif index == 12 : opname = "dilsepblock_7x7"
            elif index == 13 : opname = "skip_connect"   
                
            channels_change = blockgenotype[1]
            activation = blockgenotype[2]
            
            if blockgenotype[3] == 0: 
                pooling_or_not = False
                poolmode="None"
            elif blockgenotype[3] == 1:
                pooling_or_not = True
                poolmode="avg"
            elif blockgenotype[3] == 2:
                pooling_or_not = True
                poolmode="max"
            affine = True
            track_running_stats =True
            block = BLOCKCHOSE[opname](channels, channels_change, activation, pooling_or_not, poolmode, affine, track_running_stats)
            self.blocks.append(block)
            count = count+1
            channels = block.out_dim
            #out_channels.append(channels)
            if count == num_Sblock:  
                downsample_indim = C
                downsample_outdim = channels
                #print(downsample_indim)
                #print(downsample_outdim)
                if downsample_indim != downsample_outdim:
                    #print('use downsample')
                    downsample = DownSampleblock(downsample_indim,downsample_outdim,activation)
                    self.downsample_op.append(downsample)
                    self.downsample_count.append(1)
                else:
                    #print('dont use downsample')
                    self.downsample_count.append(0)
                downsample_indim = downsample_outdim
            elif count > num_Sblock :
                if (count-num_Sblock)%num_Dblocks == 0:
                    downsample_outdim = channels
                    #print(downsample_indim)
                    #print(downsample_outdim)
                    if downsample_indim != downsample_outdim:
                        #print('use downsample')
                        downsample = DownSampleblock(downsample_indim,downsample_outdim,activation)
                        self.downsample_op.append(downsample)
                        self.downsample_count.append(1)
                    else:
                        #print('dont use downsample')
                        self.downsample_count.append(0)
                    downsample_indim = downsample_outdim
                



        self.lastact = nn.Sequential(
            nn.BatchNorm2d(channels), 
            nn.ReLU(inplace=True)
        )
        self.feature_pooling = nn.AvgPool2d(3)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.3)
        self.classifier = nn.Linear(channels, num_classes)


    def forward(self, inputs):
        feature = self.stem(inputs)
        count = 0
        rescount=0
        feature_residual = feature
        for blockop in self.blocks:
            feature = blockop(feature)
            count = count + 1 
            if count == self.num_Sblock:
                if self.resnet_or_not_mulit[rescount] == 1:
                    if self.downsample_count[rescount] == 1 :
                        residual = self.downsample_op[rescount](feature_residual)
                        feature = feature + residual
                        feature_residual = feature
                        rescount = rescount + 1
                    elif self.downsample_count[rescount] == 0 :
                        feature = feature + feature_residual
                        feature_residual = feature
                        rescount = rescount + 1
         #       out1 = self.feature_pooling(feature)
            elif count > self.num_Sblock:
                if (count - self.num_Sblock) % self.num_Dblocks == 0:
                    if self.resnet_or_not_mulit[rescount] == 1:
                        if self.downsample_count[rescount] == 1 :
                            residual = self.downsample_op[rescount](feature_residual)
                            feature = feature + residual
                            feature_residual = feature
                            rescount = rescount + 1                       
                        elif self.downsample_count[rescount] == 0 :
                            feature = feature + feature_residual 
                            feature_residual = feature
                            rescount = rescount + 1     
        #out2 = self.feature_pooling(feature)              

        out = self.lastact(feature)
        out = self.global_pooling(out)
        out = out.view(out.size(0), -1)
        #out = self.dropout(out)
        logits = self.classifier(out)

     #   return out1, out2, logits
        return logits