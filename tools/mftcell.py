from copy import deepcopy

import torch
import torch.nn as nn

from .mftoperation import OPS

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

def op2name(operation):
    if operation == 0 : opname = "none"
    if operation == 1 : opname = "avg_pool_3x3"
    if operation == 2 : opname = "max_pool_3x3"
    if operation == 3 : opname = "nor_conv_7x7"
    if operation == 4 : opname = "nor_conv_3x3"
    if operation == 5 : opname = "nor_conv_1x1"
    if operation == 6 : opname = "dua_sepc_3x3"
    if operation == 7 : opname = "dua_sepc_5x5"
    if operation == 8 : opname = "dil_sepc_3x3"
    if operation == 9 : opname = "dil_sepc_5x5"
    if operation == 10 : opname = "spa_sec_3"
    if operation == 11 : opname = "spa_sec_5"
    if operation == 12 : opname = "skip_connect"
    return opname

# Cell for mft-dual-model
class BasciCell(nn.Module):
    def __init__(
        self, genotype,  C_in, C_out, stride, affine=True, track_running_stats=True
       ):
        super().__init__()
        self.layers = nn.ModuleList()
        #self.node_IN = []
        #self.node_IX = []
        self.genotype = deepcopy(genotype)
        genotype_length = len(self.genotype)
        
        self.status=self.genotype[:genotype_length//2]
        #print(self.status)
        self.op=self.genotype[genotype_length//2:]
        #print(self.op)
        
        for operation in self.op:
            if operation == 0 : opname = "none"
            if operation == 1 : opname = "avg_pool_3x3"
            if operation == 2 : opname = "max_pool_3x3"
            if operation == 3 : opname = "nor_conv_7x7"
            if operation == 4 : opname = "nor_conv_3x3"
            if operation == 5 : opname = "nor_conv_1x1"
            if operation == 6 : opname = "dua_sepc_3x3"
            if operation == 7 : opname = "dua_sepc_5x5"
            if operation == 8 : opname = "dil_sepc_3x3"
            if operation == 9 : opname = "dil_sepc_5x5"
            if operation == 10 : opname = "spa_sec_3"
            if operation == 11 : opname = "spa_sec_5"
            if operation == 12 : opname = "skip_connect"
            layer = OPS[opname](C_in, C_out, stride, affine, track_running_stats)
            self.layers.append(layer)
        #self.nodes = len(genotype)
        self.in_dim = C_in
        self.out_dim = C_out

    def forward(self, inputs):
        statu = []
        statu.append(inputs)
        output = []
        #nodes = [inputs]
        for index, op in zip(self.status,self.layers):
            state = op(statu[index])
            statu.append(state)
        for index in range(0,len(self.status)):
            if index in self.status : continue
            output.append(statu[index]) 
            
        output_ferature=sum(output)
        
        return output_ferature


# Cell for mft-dual-model
class BascidualCell(nn.Module):
    
    def __init__(
        self, genotype,  C_in, C_out, stride, affine=True, track_running_stats=True
       ):
        super().__init__()
        #self.layers = nn.ModuleList()
        self.in_dim = C_in
        self.out_dim = C_out

        self.genotype = deepcopy(genotype)
        genotype_length = len(self.genotype)
        self.steps = genotype_length//4
        
        #self.status=self.genotype[:genotype_length//2]
        #self.op=self.genotype[genotype_length//2:]
        
        self.nodes = []  # state
        self.ops = nn.ModuleList()  # 卷积各种操作
        self.combs = nn.ModuleList()  # 联和
        
        for i in range(0,int(len(self.genotype)/4)):
            
            n1=self.genotype[i*4+1]
            n2=self.genotype[i * 4 + 3]
            
            self.nodes.append(n1)
            self.nodes.append(n2)
            
            op1_index=self.genotype[i*4]
            op2_index = self.genotype[i * 4+2]
            op1_name = op2name(op1_index)
            op2_name = op2name(op2_index)
            
            op1 = OPS[op1_name](C_in, C_out, stride, affine, track_running_stats)
            if op1_index<3: 
                op1 = nn.Sequential(op1, nn.BatchNorm2d(C_out, affine=False))
            op2 = OPS[op2_name](C_in, C_out, stride, affine, track_running_stats)
            if op2_index<3:
                op2 = nn.Sequential(op2, nn.BatchNorm2d(C_out, affine=False))
                
            if n1==n2: #comb_name == 'add':
                self.combs.append(None)
            else:
                self.combs.append(ReLUConvBN(C_out * 2, C_out, 1, 1, 0))
                
            self.ops.append(op1)
            self.ops.append(op2) 
            
        #self.nodes = len(genotype)
        #self.preprocess = ReLUConvBN(C_in, C_in, 1, 1, 0)
           
                
    def forward(self, s0, s1):
        #s0 = self.preprocess(s0)
        #s1 = self.preprocess(s1)
        
        states = [s0, s1]
        output = []
        
        for i in range(self.steps):  #有几个基因           
            h1 = states[self.nodes[2 * i]]
            h2 = states[self.nodes[2 * i + 1]]
            op1 = self.ops[2 * i]
            op2 = self.ops[2 * i + 1]           
            h1 = op1(h1)
            #print(('h1'),h1.size())
            h2 = op2(h2)
            #print(('h2'),h2.size())
            comb = self.combs[i]
            if comb == None:
                s = h1 + h2
            else:
                s = torch.cat([h1, h2], dim=1)
                s = comb(s)
            #print(('s'),s.size()) 
            #states.append(s)
            states += [s]
            #print(('states:'),len(states))
        for index in range(0,len(states)):
            if index in self.nodes : continue
            output.append(states[index]) 
        #print(len(output))    
        output_ferature=sum(output)
        
        return output_ferature

# Cell for NAS-Bench-201
class InferCell(nn.Module):
    def __init__(
        self, genotype, C_in, C_out, stride, affine=True, track_running_stats=True
    ):
        super().__init__()

        self.layers = nn.ModuleList()
        self.node_IN = []
        self.node_IX = []
        self.genotype = deepcopy(genotype)
        for i in range(1, len(genotype)):
            node_info = genotype[i - 1]
            cur_index = []
            cur_innod = []
            for (op_name, op_in) in node_info:
                if op_in == 0:
                    layer = OPS[op_name](C_in, C_out, stride, affine, track_running_stats)
                else:
                    layer = OPS[op_name](C_out, C_out, 1, affine, track_running_stats)
                cur_index.append(len(self.layers))
                cur_innod.append(op_in)
                self.layers.append(layer)
            self.node_IX.append(cur_index)
            self.node_IN.append(cur_innod)
        self.nodes = len(genotype)
        self.in_dim = C_in
        self.out_dim = C_out

    def extra_repr(self):
        string = "info :: nodes={nodes}, inC={in_dim}, outC={out_dim}".format(
            **self.__dict__
        )
        laystr = []
        for i, (node_layers, node_innods) in enumerate(zip(self.node_IX, self.node_IN)):
            y = [f"I{_ii}-L{_il}" for _il, _ii in zip(node_layers, node_innods)]
            x = "{:}<-({:})".format(i + 1, ",".join(y))
            laystr.append(x)
        return (
            string + ", [{:}]".format(" | ".join(laystr)) + f", {self.genotype.tostr()}"
        )

    def forward(self, inputs):
        nodes = [inputs]
        for node_layers, node_innods in zip(self.node_IX, self.node_IN):
            node_feature = sum(
                self.layers[_il](nodes[_ii]) for _il, _ii in zip(node_layers, node_innods)
            )
            nodes.append(node_feature)
        return nodes[-1]


# Learning Transferable Architectures for Scalable Image Recognition, CVPR 2018
class NASNetInferCell(nn.Module):
    def __init__(
        self,
        genotype,
        C_prev_prev,
        C_prev,
        C,
        reduction,
        reduction_prev,
        affine,
        track_running_stats,
    ):
        super().__init__()
        self.reduction = reduction
        if reduction_prev:
            self.preprocess0 = OPS["skip_connect"](
                C_prev_prev, C, 2, affine, track_running_stats
            )
        else:
            self.preprocess0 = OPS["nor_conv_1x1"](
                C_prev_prev, C, 1, affine, track_running_stats
            )
        self.preprocess1 = OPS["nor_conv_1x1"](C_prev, C, 1, affine, track_running_stats)

        if not reduction:
            nodes, concats = genotype["normal"], genotype["normal_concat"]
        else:
            nodes, concats = genotype["reduce"], genotype["reduce_concat"]
        self._multiplier = len(concats)
        self._concats = concats
        self._steps = len(nodes)
        self._nodes = nodes
        self.edges = nn.ModuleDict()
        for i, node in enumerate(nodes):
            for in_node in node:
                name, j = in_node[0], in_node[1]
                stride = 2 if reduction and j < 2 else 1
                node_str = f"{i + 2}<-{j}"
                self.edges[node_str] = OPS[name](
                    C, C, stride, affine, track_running_stats
                )

    # [TODO] to support drop_prob in this function..
    def forward(self, s0, s1, unused_drop_prob):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i, node in enumerate(self._nodes):
            clist = []
            for in_node in node:
                _, j = in_node[0], in_node[1]
                node_str = f"{i + 2}<-{j}"
                op = self.edges[node_str]
                clist.append(op(states[j]))
            states.append(sum(clist))
        return torch.cat([states[x] for x in self._concats], dim=1)


class AuxiliaryHeadCIFAR(nn.Module):
    def __init__(self, C, num_classes):
        """assuming input size 8x8"""
        super().__init__()
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(
                5, stride=3, padding=0, count_include_pad=False
            ),  # image size = 2 x 2
            nn.Conv2d(C, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x