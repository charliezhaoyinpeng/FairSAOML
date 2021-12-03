import torch
import torch.nn as nn
import torch.nn.functional as F

class NN(nn.Module):
    def __init__(self, d_in):
        super(NN, self).__init__()
        self.linear1 = nn.Linear(d_in, 40)
        self.linear2 = nn.Linear(40, 40)
        self.linear3 = nn.Linear(40, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        return torch.sigmoid(x)

    def parameterised(self, x, weights):
        # like forward, but uses ``weights`` instead of ``model.parameters()``
        # it'd be nice if this could be generated automatically for any nn.Module...
        x = nn.functional.linear(x, weights[0], weights[1])
        x = nn.functional.relu(x)
        x = nn.functional.linear(x, weights[2], weights[3])
        x = nn.functional.relu(x)
        x = nn.functional.linear(x, weights[4], weights[5])
        return torch.sigmoid(x)

    def assign(self, weights):
        with torch.no_grad():
            self.linear1.weight = weights[0]
            self.linear1.bias = weights[1]
            self.linear2.weight = weights[2]
            self.linear2.bias = weights[3]
            self.linear3.weight = weights[4]
            self.linear3.bias = weights[5]

    def e_norm(self, weights):
        out = 0
        for weight in weights:
            out += torch.norm(weight)
        return out