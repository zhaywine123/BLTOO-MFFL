import torch.nn as nn


from .mftcell import BasciCell, BascidualCell
from .mftblock import Basciblock
from .mftoperation import ResNetBasicblock

class Mft_first_network(nn.Module):
    def __init__(self, C, genotype, num_layer,num_classes):
        super().__init__()
        self._C = C
        self.num = num_layer + 1
        self.stem = nn.Sequential(
            nn.Conv2d(3, C, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(C)
        )

        layer_channels = [C] * num_layer + [C * 2] 
        layer_reductions = [False] * num_layer + [True]    
        
        C_prev = C
        self.cells = nn.ModuleList()        
        for C_curr, reduction in zip(layer_channels, layer_reductions):
            if reduction:
                cell = ResNetBasicblock(C_prev, C_curr, 2, True)
            else:
                cell = BascidualCell(genotype, C_prev, C_curr, 1) 
            self.cells.append(cell) 
            C_prev = cell.out_dim
            
        self._Layer = len(self.cells)
        self.lastact = nn.Sequential(nn.BatchNorm2d(C_prev), nn.ReLU(inplace=True))
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)
        
    def forward(self, inputs):
        feature0 = feature1 = self.stem(inputs)
        for cell in self.cells:
            feature0, feature1 = feature1, cell(feature0, feature1)
        feature_b1 = feature1     
        out = self.lastact(feature1)
        out = self.global_pooling(out)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)

        return feature_b1,logits
    
    
class Mft_second_network(nn.Module):
    def __init__(self, C, genotype, num_layer,num_classes):
        super().__init__()
        self._C = C
        self.num = num_layer + 1
        
        layer_channels = [C * 2] * num_layer + [C * 4] 
        layer_reductions = [False] * num_layer + [True]        
        C_prev = C * 2
        
        self.cells = nn.ModuleList()
        for C_curr, reduction in zip(layer_channels, layer_reductions):
            if reduction:
                cell = ResNetBasicblock(C_prev, C_curr, 2, True)
            else:
                cell = BascidualCell(genotype, C_prev, C_curr, 1)
            self.cells.append(cell)
            C_prev = cell.out_dim
        self._Layer = len(self.cells)

        self.lastact = nn.Sequential(nn.BatchNorm2d(C_prev), nn.ReLU(inplace=True))
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)
        
    def forward(self, inputs):
        feature0 = feature1 = inputs
        #cell_count = 0
        for cell in self.cells:
            #print(cell_count)
            #cell_count = cell_count+1
            feature0, feature1 = feature1, cell(feature0, feature1)
        feature_b2 = feature1    
        out = self.lastact(feature1)
        out = self.global_pooling(out)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)       
        return feature_b2, logits

class Mft_classify_network(nn.Module):
    def __init__(self, C, genotype, num_layer,num_classes):
        super().__init__()
        self._C = C
        #self._layerN = N
        self.num_S = num_layer + 1        
        layer_channels = [C * 4] * num_layer + [C * 4] 
        layer_reductions = [False] * num_layer +  [True]      
        C_prev = C * 4
        
        self.cells = nn.ModuleList()
        count_layer=1
        for C_curr, reduction in zip(layer_channels, layer_reductions):
            if reduction:
                cell = ResNetBasicblock(C_prev, C_curr, 2, True)
            else:
                cell = BascidualCell(genotype, C_prev, C_curr, 1)
            count_layer = count_layer + 1    
            self.cells.append(cell)
            C_prev = cell.out_dim
        self._Layer = len(self.cells)

        self.lastact = nn.Sequential(nn.BatchNorm2d(C_prev), nn.ReLU(inplace=True))
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)
        
    def forward(self, inputs):
        feature0 = feature1 = inputs
        for cell in self.cells:
            feature0, feature1 = feature1, cell(feature0, feature1)
        out = self.lastact(feature1)
        out = self.global_pooling(out)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)
        return logits       

class Mft_network_combined(nn.Module):
    def __init__(self, model1, model2, model3):
        super(Mft_network_combined,self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3

    def forward(self, inputs):
        inputs, _ = self.model1(inputs)
        inputs, _ = self.model2(inputs)
        logits = self.model3(inputs)
        return logits       
