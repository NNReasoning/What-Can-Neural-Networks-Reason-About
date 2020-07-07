import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


class MLP(nn.Module):
    def __init__(self, n_layer, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()

        self.n_layer = n_layer 
        self.linears = nn.ModuleList([nn.Linear(input_dim, hidden_dim)])
        for layer in range(n_layer-2):
            self.linears.append(nn.Linear(hidden_dim, hidden_dim))
        self.linears.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        for layer in range(self.n_layer):
            x = self.linears[layer](x)
            x = F.relu(x)

        return x

class FCOutputModel(nn.Module):
    def __init__(self, n_layer, input_dim, hidden_dim, output_dim):
        super(FCOutputModel, self).__init__()
        self.n_layer = n_layer 
        self.linears = nn.ModuleList([nn.Linear(input_dim, hidden_dim)])
        for layer in range(n_layer-2):
            self.linears.append(nn.Linear(hidden_dim, hidden_dim))
        print(hidden_dim, output_dim)
        self.linears.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        for layer in range(self.n_layer-1):
            if layer == self.n_layer - 2:
                x = F.dropout(x)
            x = self.linears[layer](x)
            x = F.relu(x)

        x = F.dropout(x)
        x = self.linears[self.n_layer-1](x)

        return F.log_softmax(x)

class Bottleneck(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Bottleneck, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(input_dim, hidden_dim), nn.Linear(hidden_dim, output_dim)])
        
    def forward(self, x):
        x_input = x
        x = self.linears[0](x)
        x = self.linears[1](x)
        x = x_input + x
        
        return x

class FCOutputModel_SkipConnection(nn.Module):
    def __init__(self, n_layer, input_dim, hidden_dim, output_dim, block=Bottleneck):
        super(FCOutputModel_SkipConnection, self).__init__()
        self.n_layer = n_layer 
        self.blocks = torch.nn.ModuleList()
        self.blocks.append(nn.Linear(input_dim, hidden_dim))
        for layer in range(0, self.n_layer-2, 2):
            self.blocks.append(block(hidden_dim, hidden_dim, hidden_dim))
        self.blocks.append(nn.Linear(hidden_dim, output_dim)) 

    def forward(self, x):
        index = 0
        x = self.blocks[index](x)
        index = index + 1
        
        for layer in range(0, self.n_layer-2, 2):
            if layer == (self.n_layer - 4):
                x = F.dropout(x)
            x = self.blocks[index](x)
            x = F.relu(x)
            index = index + 1

        x = F.dropout(x)
        x = self.blocks[index](x)

        return F.log_softmax(x)
