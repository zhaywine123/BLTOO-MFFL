import torch.nn as nn


from .mftcell import BasciCell, BascidualCell
from .mftblock import Basciblock
from .mftoperation import ResNetBasicblock

def decoder(genotype):
        channels = genotype[0]
        channels_change = genotype[0]
        cnn_layers = genotype[1]
        cnn_kernel_size = genotype[2]
        activation = genotype[3]
        pool_or_not = genotype[4]
        pooling = genotype[5]
        stride = genotype[6]
        padding = genotype[7]
        dilation = genotype[8]
        
        if cnn_kernel_size == 1 : 
            cnn_kernel_size=1
            padding = 0
            dilation = 1
        elif  cnn_kernel_size == 2 : 
            cnn_kernel_size=3
            dilation=1
            padding = 1
        elif  cnn_kernel_size == 3 : 
            cnn_kernel_size=5
            padding = 2
            dilation = 1
        elif  cnn_kernel_size == 4 : 
            cnn_kernel_size=7
            padding = 3
            dilation = 1

        
        return channels_change,cnn_layers,cnn_kernel_size,activation,pool_or_not,pooling,stride,padding,dilation
        



class MftNetwork_block(nn.Module):
    def __init__(self, C, Sgenotype, Dgenotype, DDgenotype, num_classes):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, C, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(C)
        )
        
        affine = True
        (Schannels_change,Scnn_layers,Scnn_kernel_size,Sactivation,Spool_or_not,Spooling,Sstride,Spadding,Sdilation) = decoder(Sgenotype)
        
        (Dchannels_change,Dcnn_layers,Dcnn_kernel_size,Dactivation,Dpool_or_not,Dpooling,Dstride,Dpadding,Ddilation) = decoder(Dgenotype)
        
            
        (DDchannels_change,DDcnn_layers,DDcnn_kernel_size,DDactivation,DDpool_or_not,DDpooling,DDstride,DDpadding,DDdilation) = decoder(DDgenotype)
        
        Schannels = C
        Sblock = Basciblock(
                Schannels,
                Schannels_change,
                Sstride,
                Spadding,
                Sdilation,
                affine,
                Scnn_layers,
                Scnn_kernel_size,
                Sactivation,
                Spool_or_not,
                Spooling)
        
        Dchannels = Sblock.out_dim
        Dblock = Basciblock(
                Dchannels,
                Dchannels_change,
                Dstride,
                Dpadding,
                Ddilation,
                affine,
                Dcnn_layers,
                Dcnn_kernel_size,
                Dactivation,
                Dpool_or_not,
                Dpooling)
        
        DDchannels = Dblock.out_dim
        DDblock = Basciblock(
                DDchannels,
                DDchannels_change,
                DDstride,
                DDpadding,
                DDdilation,
                affine,
                DDcnn_layers,
                DDcnn_kernel_size,
                DDactivation,
                DDpool_or_not,
                DDpooling)
        
        C_out = DDblock.out_dim
        
        self.blocks = nn.ModuleList()
        self.blocks.append(Sblock)
        self.blocks.append(Dblock)
        self.blocks.append(DDblock)
        self.lastact = nn.Sequential(nn.BatchNorm2d(C_out), nn.ReLU(inplace=True))
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_out, num_classes)


    def forward(self, inputs):
        feature = self.stem(inputs)
        count = 0
        for block in self.blocks:
            feature = block(feature)
            count=count+1
            if count==1: out1 = feature
            if count==3: out2 = feature

        out = self.lastact(feature)
        out = self.global_pooling(out)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)

        return out1, out2,logits



class MftNetwork(nn.Module):
    def __init__(self, C, Sgenotype, Dgenotype, num_Slayer, num_Dlayer, num_DDlayer ,num_classes):
        super().__init__()
        self._C = C
        #self._layerN = N
        self.num_S = num_Slayer + 1
        self.num_d = num_Slayer + 1 + num_Dlayer + 1
        self.num_dd = num_Slayer + 1 + num_Dlayer + 1 + num_DDlayer
        self.stem = nn.Sequential(
            nn.Conv2d(3, C, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(C)
        )

        layer_channels = [C] * num_Slayer + [C * 2] + [C * 2] * num_Dlayer + [C * 4] + [C * 4] * num_DDlayer
        layer_reductions = [False] * num_Slayer + [True] + [False] * num_Dlayer + [True] + [False] * num_DDlayer
        
        #layer_channels = [C] * num_Slayer + [C * 2] + [C * 2] * num_Dlayer + [C * 4] 
        #layer_reductions = [False] * num_Slayer + [True] + [False] * num_Dlayer + [True] 
        C_prev = C
        self.cells = nn.ModuleList()
        count_layer=1
        for C_curr, reduction in zip(layer_channels, layer_reductions):
            if count_layer <= num_Slayer + 1: 
                genotype = Sgenotype
            elif count_layer <= num_Slayer + 1 + num_Dlayer:
                genotype = Dgenotype
            elif count_layer <= num_Slayer + 1 + num_Dlayer + 1 + num_DDlayer:
                genotype = Dgenotype
                
            if reduction:
                cell = ResNetBasicblock(C_prev, C_curr, 2, True)
            else:
                cell = BasciCell(genotype, C_prev, C_curr, 1)
            count_layer = count_layer + 1    
            self.cells.append(cell)
            C_prev = cell.out_dim
        self._Layer = len(self.cells)

        self.lastact = nn.Sequential(nn.BatchNorm2d(C_prev), nn.ReLU(inplace=True))
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)


    def forward(self, inputs):
        feature = self.stem(inputs)
        count_cell = 0
        for cell in self.cells:
            count_cell = count_cell + 1
            feature = cell(feature)
            if count_cell == (self.num_S) :
                out1 = self.global_pooling(feature)
            if count_cell == (self.num_d ):
                out2 = self.global_pooling(feature)

        out = self.lastact(feature)
        out = self.global_pooling(out)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)

        return out1, out2,logits
    
class MftNetwork_dual(nn.Module):
    def __init__(self, C, Sgenotype, Dgenotype, DDgenotype, num_Slayer, num_Dlayer, num_DDlayer ,num_classes):
        super().__init__()
        self._C = C
        #self._layerN = N
        self.num_S = num_Slayer + 1
        self.num_d = num_Slayer + 1 + num_Dlayer + 1
        self.num_dd = num_Slayer + 1 + num_Dlayer + 1 + num_DDlayer
        self.stem = nn.Sequential(
            nn.Conv2d(3, C, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(C)
        )

        layer_channels = [C] * num_Slayer + [C * 2]  + [C * 2] * num_Dlayer + [C * 4]  + [C * 4] * num_DDlayer
        layer_reductions = [False] * num_Slayer + [True]  + [False] * num_Dlayer + [True]  + [False] * num_DDlayer
        
        #layer_channels = [C] * num_Slayer + [C * 2] + [C * 2] * num_Dlayer + [C * 4] 
        #layer_reductions = [False] * num_Slayer + [True] + [False] * num_Dlayer + [True] 
        C_prev = C
        self.cells = nn.ModuleList()
        count_layer=1
        for C_curr, reduction in zip(layer_channels, layer_reductions):
            if count_layer <= num_Slayer + 1: 
                genotype = Sgenotype
            elif count_layer <= num_Slayer + 1 + num_Dlayer:
                genotype = Dgenotype
            elif count_layer <= num_Slayer + 1 + num_Dlayer + 1 + num_DDlayer:
                genotype = DDgenotype
                
            if reduction:
                cell = ResNetBasicblock(C_prev, C_curr, 2, True)
                self.cells.append(cell)
                self.cells.append(cell)
            else:
                cell = BascidualCell(genotype, C_prev, C_curr, 1)
                self.cells.append(cell)
            count_layer = count_layer + 1    
            C_prev = cell.out_dim
            
        self._Layer = len(self.cells)
        self.feature_pooling = nn.AvgPool2d(3)
        self.lastact = nn.Sequential(nn.BatchNorm2d(C_prev), nn.ReLU(inplace=True))
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)


    def forward(self, inputs):
        #feature = self.stem(inputs)
        feature0 = feature1 = self.stem(inputs)
        count_cell = 0
        for cell in self.cells:
            #print('count:',count_cell)
            feature0, feature1 = feature1, cell(feature0, feature1)
            #print(feature0.size())
            #print(feature1.size())
            #feature = cell(feature)
            if count_cell == (self.num_S) :
                #out1 = self.feature_pooling(feature1)
                out1 = feature1
            if count_cell == (self.num_d):
                #out2 = self.feature_pooling(feature1)
                out2 = feature1
            count_cell = count_cell + 1

        out = self.lastact(feature1)
        out = self.global_pooling(out)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)

        return out1, out2,logits