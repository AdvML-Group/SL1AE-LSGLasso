import torch
import torch.nn as nn
import torch.nn.functional as F

import math


class sReLU(nn.Module):
    """
        Smoothed Rectified Linear Unit:
            sReLU(x) = 1/s * log(1 + exp(s*x)).

        Issues:
            [Numerical Instability] The original computation "1.0 / self.s * torch.log(1 + torch.exp(self.s * x))" in forward() will lead to nan value in loss. The reason is that: if "s*x" is big enough, "exp(s*x)" will explode.
    """

    def __init__(self, s=10.0, t=10.0):
        super(sReLU, self).__init__()
        self.s = s
        self.t = t

    def forward(self, x):
        """
                |--- x          if x > t / s,
            y = |
                |--- sReLU(x)   otherwise.
        """
        x_big = F.threshold(x, self.t / self.s, 0)  # s*x>t
        x_small = x - x_big
        offset = torch.where(x > self.t / self.s, 1.0 / self.s * math.log(2.0), 0.0)  # 1/s*log(1+exp(s*0))=1/s*log(2)
        return x_big + 1.0 / self.s * torch.log(1 + torch.exp(self.s * x_small)) - offset


class ScaledTanh(nn.Module):
    def __init__(self):
        super(ScaledTanh, self).__init__()

    def forward(self, input):
        return 0.5 * (torch.tanh(input) + 1)

class AutoEncoder(nn.Module):
    """
        Deep Data Reconstruction Using L1-AutoEncoder:
            min_{W} ||X - X_||_1 + lambda * ||W||_F^2
    """

    def __init__(self, nlayers, activation, xavier_init=True):
        super(AutoEncoder, self).__init__()
        self.nlayers = nlayers
        self.act = activation
        self.encoder_layers = self.init_encoder_layers()
        self.decoder_layers = self.init_decoder_layers()
        if xavier_init:
            self.init_parameters()

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def encoder(self, x):
        for layer in self.encoder_layers:
            x = self.act(layer(x))
        return x

    def decoder(self, x):
        for layer in self.decoder_layers:
            x = self.act(layer(x))
        return x

    def init_encoder_layers(self, ):
        return nn.ModuleList([nn.Linear(nin, nout) for nin, nout in zip(self.nlayers[:-1], self.nlayers[1:])])

    def init_decoder_layers(self, ):
        return nn.ModuleList([nn.Linear(nin, nout) for nin, nout in zip(self.nlayers[-1:0:-1], self.nlayers[-2::-1])])

    def init_parameters(self, ):
        """
            Xavier's Initialization:
                W ~ U[-r, r], r = sqrt(6)/sqrt(n_{j} + n_{j+1})
        """
        all_model_layers = [layer for layer in self.encoder_layers] + [layer for layer in self.decoder_layers]
        for layer in all_model_layers:
            fan_in, fan_out = layer.weight.shape
            bound = math.sqrt(6) / math.sqrt(fan_in + fan_out)
            nn.init.uniform_(layer.weight, -bound, bound)
            nn.init.uniform_(layer.bias, -bound, bound)

    def print_parameters(self, ):
        all_model_layers = [layer for layer in self.encoder_layers] + [layer for layer in self.decoder_layers]
        for i, layer in enumerate(all_model_layers):
            print('%d-th layer: %d => %d' % (i + 1, layer.weight.shape[1], layer.weight.shape[0]))
            print('weight: ', layer.weight.data)
            print('bias: ', layer.bias.data)
            print(10 * '-')
