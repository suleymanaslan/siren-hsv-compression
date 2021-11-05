import torch
import numpy as np
import torch.nn as nn

class FCBlock(nn.Module):
    def __init__(self, in_features, out_features, num_hidden_layers, hidden_features):
        super().__init__()

        self.net = []
        self.net.append(nn.Sequential(nn.Linear(in_features, hidden_features), Sine()))
        for i in range(num_hidden_layers):
            self.net.append(nn.Sequential(nn.Linear(hidden_features, hidden_features), Sine()))
        self.net.append(nn.Sequential(nn.Linear(hidden_features, out_features)))

        self.net = nn.Sequential(*self.net)
        self.net.apply(sine_init)
        self.net[0].apply(first_layer_sine_init)
    
    def forward(self, inputs):
        return self.net(inputs)

class FCBlockMulti(nn.Module):
    def __init__(self, in_features, out_features, num_hidden_layers, hidden_features):
        super().__init__()

        self.net = []
        self.net.append(nn.Sequential(nn.Linear(in_features, hidden_features), SineMulti()))
        for i in range(num_hidden_layers):
            self.net.append(nn.Sequential(nn.Linear(hidden_features*2, hidden_features), SineMulti()))
        self.net.append(nn.Sequential(nn.Linear(hidden_features*2, out_features)))

        self.net = nn.Sequential(*self.net)
        self.net.apply(sine_init)
        self.net[0].apply(first_layer_sine_init)
    
    def forward(self, inputs):
        return self.net(inputs)
    
def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)

def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1 / num_input, 1 / num_input)

class Sine(nn.Module):
    def __init(self):
        super().__init__()

    def forward(self, input):
        return torch.sin(30 * input)
    
class SineMulti(nn.Module):
    def __init(self):
        super().__init__()

    def forward(self, input):
        return torch.cat((torch.sin(30 * input), 
                          torch.cos(30 * input)), dim=-1)

class Siren(nn.Module):
    def __init__(self, in_features, out_features, hidden_features, num_hidden_layers):
        super().__init__()
        self.net = FCBlock(in_features, out_features, num_hidden_layers, hidden_features)

    def forward(self, inputs):
        return self.net(inputs)

class SirenMulti(nn.Module):
    def __init__(self, in_features, out_features, hidden_features, num_hidden_layers):
        super().__init__()
        self.net = FCBlockMulti(in_features, out_features, num_hidden_layers, hidden_features)

    def forward(self, inputs):
        return self.net(inputs)
