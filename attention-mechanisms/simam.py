""" 
PyTorch implementation of SimAM: A Simple, Parameter-Free Attention Module for Convolutional Neural Networks

As described in http://proceedings.mlr.press/v139/yang21o/yang21o.pdf

SimAM, inspired by neuroscience theories in the mammalian brain.
"""




import torch
from torch import nn


class simam_module(torch.nn.Module):
    def __init__(self, channels = None, e_lambda = 1e-4):
        super(simam_module, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):

        b, c, h, w = x.size()
        
        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2,3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2,3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)

if __name__ == "__main__":
    x = torch.randn(2, 64, 32, 32)
    attn = simam_module()
    y = attn(x)
    print(y.shape)