
import torch.nn as nn
import re
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

OPS=[
lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1), #'max_pool_3x3' :
lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False), #ave_pool_3x3
lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine), #'identity' :
lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),  #'sep_conv_3x3':
lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),  #'sep_conv_5x5' :
lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),   #   'dil_conv_3x3':
lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),   # 'dil_conv_5x5':
lambda C, stride, affine: DilConv(C, C, 7, stride, 6, 2, affine=affine),  #  'dil_conv_7x7':
lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),  #  'sep_conv_7x7':
lambda C, stride, affine: nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv2d(C, C, (1, 3), stride=(1, stride), padding=(0, 1), bias=False),
        nn.Conv2d(C, C, (3, 1), stride=(stride, 1), padding=(1, 0), bias=False),
        nn.BatchNorm2d(C, affine=affine)
    ),#[3*1,1*3]
lambda C, stride, affine: nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv2d(C, C, (1, 5), stride=(1, stride), padding=(0, 2), bias=False),
        nn.Conv2d(C, C, (5, 1), stride=(stride, 1), padding=(2, 0), bias=False),
        nn.BatchNorm2d(C, affine=affine)
    ),#[5*1,1*5]
lambda C, stride, affine: nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv2d(C, C, (1, 7), stride=(1, stride), padding=(0, 3), bias=False),
        nn.Conv2d(C, C, (7, 1), stride=(stride, 1), padding=(3, 0), bias=False),
        nn.BatchNorm2d(C, affine=affine)
    )#[7x1_1x7]
]

def make_divisible(x, divisible_by=8): #划分
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)

class Hard_Sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hard_Sigmoid, self).__init__()
        self.inplace = inplace
    def forward(self, x):
        x = 0.2 * x + 0.5
        if self.inplace:
            return x.clamp_(0, 1)
        else:
            return x.clamp(0, 1)
#SE层
class SELayer(nn.Module):
    def __init__(self, num_in,**kwargs):
        super(SELayer, self).__init__(**kwargs)

        num_out = num_in
        ratio=2
        num_mid = make_divisible(num_out // ratio)
        #num_mid=num_in//16
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_in, num_mid, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_mid, num_out, 1, bias=True),
            Hard_Sigmoid()
        )

    def forward(self, x):
        out = self.channel_attention(x)
        return x * out
#     def __init__(self, channel, reduction=16):
#         super(SELayer, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(channel, channel // reduction, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(channel // reduction, channel, bias=False),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         print("x.size()",x.size())
#         #b, c, _, _ = x.size()  #16，24，128，128
#         y = self.avg_pool(x)#.view(b, c)
#         y = self.fc(y)#.view(b, c, 1, 1)
#         return x * y.expand_as(x)

class NLDenoise2D(nn.Module):
    def __init__(self, filter_type,inp, stride, oup=None, mid_channel=None):
                 #bn_layer=True):

        super(NLDenoise2D, self).__init__()

        assert filter_type in ['embedded_gaussian', 'gaussian']
        self.filter_type = filter_type
        self.stride=stride
        self.inp = inp
        self.mid_channel = self.inp // 2 if mid_channel is None else mid_channel
        self.oup = inp if oup is None else oup

        # g function for Non-Local Block
        self.g = nn.Conv2d(self.inp, self.mid_channel, kernel_size=1, stride=1, padding=0)
        self.W_z = nn.Conv2d(self.mid_channel, self.oup, kernel_size=1, stride=1, padding=0)
        nn.init.constant_(self.W_z.weight, 0)
        nn.init.constant_(self.W_z.bias, 0)

        #define phi and theta funciton for Non-Local Block
        if self.filter_type == 'embedded_gaussian':
            self.phi = nn.Conv2d(self.inp, self.mid_channel, kernel_size=1, stride=1, padding=0)
            self.theta = nn.Conv2d(self.inp, self.mid_channel, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # Input Shape: (Batch, Channle, Height, Weight)
        #print("non-local block里的x size",x.size())
        batch_size = x.size(0)
        # (Batch, Channel, Height*Weight)
        g_x = self.g(x).view(batch_size, self.mid_channel, -1)
        #print("self.g(x)后的x size",g_x.size())
        #(Batch, Height*Weight, Channel)
        g_x = g_x.permute(0, 2, 1)

        if self.filter_type == 'gaussian':
            theta_x = x.view(batch_size, self.inp, -1)
            phi_x = x.view(batch_size, self.inp, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)

        if self.filter_type == 'embedded_gaussian':
            theta_x = self.theta(x).view(batch_size, self.mid_channel, -1)
            phi_x = self.phi(x).view(batch_size, self.mid_channel, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)

        f_div_C = F.softmax(f, dim=-1)
        y = torch.matmul(f_div_C, g_x)

        # contiguous here just allocates contiguous chunk of memory
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.mid_channel, *x.size()[2:])

        out = self.W_z(y)
        return out


class ReLUConvBN(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(ReLUConvBN, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
      nn.BatchNorm2d(C_out, affine=affine)
    )

  def forward(self, x):
    return self.op(x)

class SepConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)

class Noisy(nn.Module):

  def __init__(self):
    super(Noisy, self).__init__()

  def forward(self, x):
    noise = torch.randn(x.shape).to(x.device) * 0.1
    return x + noise


class DilConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                      groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)

class Identity(nn.Module):

  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x):
    return x

class FactorizedReduce(nn.Module):

  def __init__(self, C_in, C_out, affine=True):
    super(FactorizedReduce, self).__init__()
    assert C_out % 2 == 0
    self.relu = nn.ReLU(inplace=False)
    self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
    self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
    self.bn = nn.BatchNorm2d(C_out, affine=affine)

  def forward(self, x):
    x = self.relu(x)
    #print("FactorizedReduce之前的通道大小",x.size())
    out = torch.cat([self.conv_1(x), self.conv_2(x[:,:,1:,1:])], dim=1)
    out = self.bn(out)
   # print("FactorizedReduce之后的通道大小",out.size())
    return out

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Cell(nn.Module):
    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev, steps=4,SE=False):
        """
        :param genotype_code:  cell的编码
        :param C_prev_prev: 
        :param C_prev:   
        :param C:  
        :param reduction:  bool 是否为reduction类型
        :param reduction_prev:  是否需要为cell进行reduction操纵
        :param steps:  每个基因组的基因数目
        """
        self.se_layer = None
        self.reduction= reduction
        super(Cell, self).__init__()
        self.steps = steps #基因的数目
        self.C = C  # 输入输出通道
        
        # preprocess0，preprocess1是cell前的两个输入
        if reduction_prev:  # reduction之前需要进行的操作
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        else:  # 不需要reduction，直接线性卷积
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)
        self.mapreduce=FactorizedReduce(C_prev_prev, C)

        
        self.geno = genotype
        self.concat= []
        for i in range(0, self.steps+2):
            if i not in self.geno[1::2]:
                self.concat.append(i)

        self.compiler(C, reduction)  # 编译
        self.multiplier = len(self.concat)  # 乘数
        if SE:
            self.se_layer = SELayer(self.multiplier * C)
        


    def compiler(self, C, reduction):
        self.nodes = []  # 节点
        self.ops = nn.ModuleList()  # 卷积各种操作
        self.combs = nn.ModuleList()  # 联和
        for i in range(0,int(len(self.geno)/4)):
            
            n1=self.geno[i*4+1]
            n2=self.geno[i * 4 + 3]
#             print(f'第{i}个节点的俩个操作节点{n1,n2}')
            self.nodes.append(n1)
            self.nodes.append(n2)
            op1_name=self.geno[i*4]
            op2_name = self.geno[i * 4+2]
#             print(f'第{i}个节点的俩个操作方式{op1_name,op2_name}')
            stride1 = 2 if reduction and n1 < 2 else 1
            op1 = OPS[op1_name](C, stride1, False)
            if op1_name<2: #如果是池化操作
                op1 = nn.Sequential(op1, nn.BatchNorm2d(C, affine=False))

            stride2 = 2 if reduction and n2 < 2 else 1
            op2 = OPS[op2_name](C, stride2, False)
            if op2_name<2: #如果是池化操作
                op2 = nn.Sequential(op2, nn.BatchNorm2d(C, affine=False))

            self.ops.append(op1)
            self.ops.append(op2)

            if n1==n2: #comb_name == 'add':
                self.combs.append(None)
            else:
                self.combs.append(ReLUConvBN(self.C * 2, self.C, 1, 1, 0))


    def forward(self, s0, s1):  
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        #print("--------每一次操作后的s的size-----")
        states = [s0, s1]
        for i in range(self.steps):  #有几个基因           
            h1 = states[self.nodes[2 * i]]
            h2 = states[self.nodes[2 * i + 1]]
            op1 = self.ops[2 * i]
            op2 = self.ops[2 * i + 1]
            h1 = op1(h1)
#             print("--------每一次操作后的s的size-----")
#             print(f'--------{ops_name(self.geno[i*4])}操作后的size{h1.size()}-----')
            h2 = op2(h2)
#             print(f'--------{ops_name(self.geno[i*4+2])}操作后的size{h2.size()}-----')
            comb = self.combs[i]
            if comb == None:
                s = h1 + h2
            else:
                s = torch.cat([h1, h2], dim=1)
                s = comb(s)
            #print(s.size())
            states += [s]
        
        #排除输入0,1被孤立的情况
        cat_=[]
        for i in self.concat:
            if i <2 and  self.reduction:
                map_reduce = self.mapreduce(s0)
                cat_.append(map_reduce) 
            else:
                cat_.append(states[i])
                
#         print("--------一个cell的最终输出连接节点------------")
#         print(self.concat)
#         for st in cat_:
#            print(states[st].size()) 
#         self.concat=cat_
#         oout=torch.cat([states[i] for i in self.concat], dim=1)
        
        oout=torch.cat(cat_,dim=1)
        if self.se_layer is None:
            return oout
        else:
            return self.se_layer(oout)
#         print("--------每一次操作后的最后hi的size-----")
#         print(oout.size())
class ResCell(nn.Module):
    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev, steps=4,SE=False):
        """
        :param genotype_code:  cell的编码
        :param C_prev_prev: 
        :param C_prev:   
        :param C:  
        :param reduction:  bool 是否为reduction类型
        :param reduction_prev:  是否需要为cell进行reduction操纵
        :param steps:  每个基因组的基因数目
        """
        self.se_layer = None
        self.reduction= reduction
        super(ResCell, self).__init__()
        self.steps = steps #基因的数目
        self.C = C  # 输入输出通道
        # preprocess0，preprocess1是cell前的两个输入
        if reduction_prev:  # reduction之前需要进行的操作
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        else:  # 不需要reduction，直接线性卷积
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)
        self.mapreduce=FactorizedReduce(C_prev_prev, C)

        
        self.geno = genotype
        self.concat= []
        for i in range(0, self.steps+2):
            if i not in self.geno[1::2]:
                self.concat.append(i)

        self.compiler(C, reduction)  # 编译
        self.multiplier = len(self.concat)  # 乘数
        if SE:
            self.se_layer = SELayer(self.multiplier * C)
            
        #self.inplanes != planes * block.expansion:
        
        self.downsample_s0 = nn.Sequential(
                conv1x1(C_prev_prev, C, 1),
                nn.BatchNorm2d(C))
        self.reduce_s0 = nn.Sequential(
                conv1x1(C_prev_prev, C, 2),
                nn.BatchNorm2d(C))
        self.downsample_s1 = nn.Sequential(
                conv1x1(C_prev, C, 1),
                nn.BatchNorm2d(C))
        self.reduce_s1 = nn.Sequential(
                conv1x1(C_prev, C, 2),
                nn.BatchNorm2d(C))
        
    def compiler(self, C, reduction):
        self.nodes = []  # 节点
        self.ops = nn.ModuleList()  # 卷积各种操作
        self.combs = nn.ModuleList()  # 联和
        for i in range(0,int(len(self.geno)/4)):
            
            n1=self.geno[i*4+1]
            n2=self.geno[i * 4 + 3]
#             print(f'第{i}个节点的俩个操作节点{n1,n2}')
            self.nodes.append(n1)
            self.nodes.append(n2)
            op1_name=self.geno[i*4]
            op2_name = self.geno[i * 4+2]
#             print(f'第{i}个节点的俩个操作方式{op1_name,op2_name}')
            stride1 = 2 if reduction and n1 < 2 else 1
    
            op1 = OPS[op1_name](C, stride1, False)
        
            if op1_name<2: #如果是池化操作
                op1 = nn.Sequential(op1, nn.BatchNorm2d(C, affine=False))

            stride2 = 2 if reduction and n2 < 2 else 1
            op2 = OPS[op2_name](C, stride2, False)
            if op2_name<2: #如果是池化操作
                op2 = nn.Sequential(op2, nn.BatchNorm2d(C, affine=False))

            self.ops.append(op1)
            self.ops.append(op2)

            if n1==n2: #comb_name == 'add':
                self.combs.append(None)
            else:
                self.combs.append(ReLUConvBN(self.C * 2, self.C, 1, 1, 0))


    def forward(self, s0, s1):  
        s0_=s0.detach()
        s1_=s1.detach()
        
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        
        
        
        identity0 = self.downsample_s0(s0_)
        identity1 = self.downsample_s1(s1_) 
        
        reduce0 = self.reduce_s0(s0_)
        reduce1 = self.reduce_s1(s1_)  
        
        #print('identity0 shape:',identity0.shape)
        #print('identity1 shape:',identity1.shape)
        #print('reduce0 shape:',reduce0.shape)
        #print('reduce1 shape:',reduce1.shape)
        #print("--------每一次操作后的s的size-----")
        
        ident0 = [identity0,identity0]
        ident1 = [identity1,identity1]
        states = [s0, s1]
        for i in range(self.steps):  #有几个基因           
            h1 = states[self.nodes[2 * i]]
            h2 = states[self.nodes[2 * i + 1]]
            op1 = self.ops[2 * i]
            op2 = self.ops[2 * i + 1]
            h1 = op1(h1)
#             print("--------每一次操作后的s的size-----")
#             print(f'--------{ops_name(self.geno[i*4])}操作后的size{h1.size()}-----')
            h2 = op2(h2)
#             print(f'--------{ops_name(self.geno[i*4+2])}操作后的size{h2.size()}-----')
            comb = self.combs[i]
            if comb == None:
                s = h1 + h2
            else:
                s = torch.cat([h1, h2], dim=1)
                s = comb(s)
            if self.reduction:
                ident0_temp = reduce0
                ident1_temp = reduce1
            else:
                ident0_temp = identity0
                ident1_temp = identity1
            #print(s.size())
            states += [s]
            ident0 += [ident0_temp]
            ident1 += [ident1_temp]
            
        #排除输入0,1被孤立的情况
        cat_=[]
        id0_=[]
        id1_=[]
        
        #print('s0 shape',s0.shape)
        #print('s1 shape',s1.shape)
        
      #  for i in range(len(states)):
      #      print(i)
      #      print('state shape',states[i].shape)
      #      print('re0 shape',ident0[i].shape)  
      #      print('re1 shape',ident1[i].shape)  
      #  print(self.concat)
        
        for i in self.concat:
            if i <2 and self.reduction:
                map_reduce = self.mapreduce(s0)
                #id0_reduce = self.mapreduce(identity0)
                #id1_reduce = self.mapreduce(identity1)
                cat_.append(map_reduce) 
                id0_.append(reduce0) 
                id1_.append(reduce1) 
            else:
                cat_.append(states[i])
                id0_.append(ident0[i]) 
                id1_.append(ident1[i]) 
                     
#         print("--------一个cell的最终输出连接节点------------")
#         print(self.concat)
#         for st in cat_:
#            print(states[st].size()) 
#         self.concat=cat_
#         oout=torch.cat([states[i] for i in self.concat], dim=1)
        #print('-----------------------------')
        #for i in range(len(cat_)):
        #    print('--------------',i)
        #    print('o',cat_[i].shape)
        #    #print('1',id0_[i].shape)
        #    print('2',id1_[i].shape)
        #print('-----------------------------')
        oout=torch.cat(cat_,dim=1)
        #id0out=torch.cat(id0_,dim=1)
        id1out=torch.cat(id1_,dim=1)
        
        #print('oout',oout.shape)
        #print('id0out',id0out.shape)
        #print('id1out',id1out.shape)
        
        #res_out = oout + id0out + id1out
        res_out = oout + id1out
        #print('res_out',res_out.shape)
        if self.se_layer is None:
            res_out = res_out
        else:
            res_out = self.se_layer(res_out) 
        
        
        return res_out
#         print("--------每一次操作后的最后hi的size-----")
#         print(oout.size())      

class Network(nn.Module):
    def __init__(self, genotype,
                 restype,
                 num_classes=4, 
                 C=32, 
                 stem_multiplier=2, 
                 layers=3,
                 SE_=True,
                 non_local=True):
        super(Network, self).__init__()

        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(C_curr)
        )
        self.relu = nn.ReLU(inplace=True)
        
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        
        for i in range(layers):  #layers 有几个normal和reduction的堆叠
            if i not in [0,2]:
                C_curr *= 2
                reduction = True
                cell_type_num=1
            else:
                reduction = False
                cell_type_num=0
            if restype[i] == 0:
                cell = Cell(genotype[cell_type_num], C_prev_prev, C_prev, C_curr, reduction, reduction_prev, steps=6,SE=SE_)
            if restype[i] == 1:
                cell = ResCell(genotype[cell_type_num], C_prev_prev, C_prev, C_curr, reduction, reduction_prev, steps=6,SE=SE_)    
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            #filter_type=[embedded_gaussian ,gaussian]
            
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.non_local=non_local
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, input):
        s0 = s1 = self.stem(input)
        #surface feature
        f1 = self.maxpool(s0)

        
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1)
        #deep feature    
        f2 = self.maxpool(s1)
        
        out = self.global_pooling(s1)
        out = torch.flatten(out, 1)
        #logits = self.classifier(out.view(out.size(0), -1))
        logits = self.classifier(out.view(out.size(0), -1))
        return f1,f2,logits