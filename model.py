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

class FCBlockEncoding(nn.Module):
    def __init__(self, in_features, out_features, num_hidden_layers, hidden_features):
        super().__init__()

        self.net = []
        self.net.append(nn.Sequential(nn.Linear(in_features, hidden_features), SinusoidalEncoding()))
        for i in range(num_hidden_layers):
            self.net.append(nn.Sequential(nn.Linear(hidden_features*12, hidden_features), SinusoidalEncoding()))
        self.net.append(nn.Sequential(nn.Linear(hidden_features*12, out_features)))

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

class SinusoidalEncoding(nn.Module):
    def __init(self):
        super().__init__()

    def forward(self, input):
        return torch.cat([torch.cat((torch.sin(2 ** (i+1) * input), torch.cos(2 ** (i+1) * input)), dim=-1) for i in range(6)], dim=-1)

class Siren(nn.Module):
    def __init__(self, in_features, out_features, hidden_features, num_hidden_layers):
        super().__init__()
        self.net = FCBlock(in_features, out_features, num_hidden_layers, hidden_features)

    def forward(self, inputs):
        return self.net(inputs)

class SirenFeatureEncoding(nn.Module):
    def __init__(self, in_features, out_features, hidden_features, num_hidden_layers):
        super().__init__()
        self.net = FCBlockEncoding(in_features, out_features, num_hidden_layers, hidden_features)

    def forward(self, inputs):
        return self.net(inputs)

class Signet(Siren):
    def __init__(self, in_features, out_features, hidden_features, num_hidden_layers, c, alpha, batch_coord):
        super().__init__(in_features * c, out_features, hidden_features, num_hidden_layers)
        self.c = c
        self.alpha = alpha
        self.in_features = in_features * self.c
        self.gegenbauer_init = torch.ones(list(batch_coord.view(-1, batch_coord.shape[-1]).shape) + [self.c])
        self.out_shape = list(batch_coord.shape)
        self.out_shape[-1] = self.out_shape[-1] * self.c
        
    def update_gegenbauer_init(self, batch_coord):
        self.gegenbauer_init = torch.ones(list(batch_coord.view(-1, batch_coord.shape[-1]).shape) + [self.c])
        self.out_shape = list(batch_coord.shape)
        self.out_shape[-1] = self.out_shape[-1] * self.c
    
    def input_transformation(self, in_coord):
        in_coord = in_coord.view(-1, in_coord.shape[-1])
        gegenbauer_polynomials = self.gegenbauer_init.clone().to(in_coord.device)
        gegenbauer_polynomials[...,1] = 2*self.alpha*in_coord
        for n in range(1, self.c-1):
            gegenbauer_polynomials[...,n+1] = (1/n) * (2*in_coord*(n+self.alpha-1)*gegenbauer_polynomials[...,n] - (n+2*self.alpha-2)*gegenbauer_polynomials[...,n-1])
        return gegenbauer_polynomials.view(self.out_shape)

    def forward(self, inputs):
        inputs = inputs.clamp(-1, 1)
        inputs_transformed = self.input_transformation(inputs)
        return self.net(inputs_transformed)
