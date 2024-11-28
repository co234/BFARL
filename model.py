import torch
import torch.nn as nn
from meta_module import MetaModule, MetaLinear



class MLP(MetaModule):
    def __init__(self, feature_dim, hidsizes, outputs=1, dropout=0.15, activation='relu'):
        super(MLP, self).__init__()

        if activation == 'relu':
            self.ac_fn = torch.nn.ReLU
        elif activation == 'tanh':
            self.ac_fn = torch.nn.Tanh
        elif activation == 'sigmoid':
            self.ac_fn = torch.nn.Sigmoid
        elif activation == 'leaky':
            self.ac_fn = torch.nn.LeakyReLU
        elif activation == 'elu':
            self.ac_fn = torch.nn.ELU
        elif activation == 'relu6':
            self.ac_fn = torch.nn.ReLU6

    
        self.mlp = []
        hidsizes = [feature_dim] + hidsizes
        for i in range(1, len(hidsizes)):
            self.mlp.append(MetaLinear(hidsizes[i-1], hidsizes[i]))
            self.mlp.append(nn.Dropout(dropout))
            self.mlp.append(self.ac_fn())
        self.mlp = nn.Sequential(*self.mlp, MetaLinear(hidsizes[-1], outputs))

    def forward(self, x):
        if type(x) != torch.Tensor:
            x = torch.tensor(x, dtype=torch.float)
        if x.dim() < 2:
            x = x.unsqueeze(0)
        if torch.cuda.is_available():
            x = x.cuda()
        return self.mlp(x).squeeze(-1)
