# https://github.com/BCV-Uniandes/Gabor_Layers_for_Robustness/tree/master
# MIT License

# Copyright (c) 2020 Biomedical Computer Vision @ Uniandes

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class GaborLayerLearnable(nn.Module):
    def __init__(self, in_channels, stride, padding, kernels, out_channels=64,
        kernel_size=11, bias1=False, bias2=False, relu=True,
        use_alphas=True, use_translation=False):
        super(GaborLayerLearnable, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.kernels = kernels
        self.total_kernels = self.kernels
        self.responses = self.kernels
        self.kernel_size = kernel_size
        self.relu = relu
        self.use_alphas = use_alphas
        self.use_translation = use_translation
        # All parameters have the same size as the number of families and input channels
        # Replicate parameters so that broadcasting is possible
        self.Lambdas = nn.Parameter(torch.randn(self.total_kernels).unsqueeze(
            dim=1).unsqueeze(dim=2))
        self.psis = nn.Parameter(torch.randn(self.total_kernels).unsqueeze(
            dim=1).unsqueeze(dim=2))
        self.sigmas = nn.Parameter(torch.randn(self.total_kernels).unsqueeze(
            dim=1).unsqueeze(dim=2))
        self.gammas_y = nn.Parameter(torch.randn(self.total_kernels).unsqueeze(
            dim=1).unsqueeze(dim=2))

        if self.use_translation:
            self.trans_x = nn.Parameter(torch.randn(self.total_kernels).unsqueeze(
                dim=1).unsqueeze(dim=2))
            self.trans_y = nn.Parameter(torch.randn(self.total_kernels).unsqueeze(
                dim=1).unsqueeze(dim=2))

        if self.use_alphas:
            self.alphas = nn.Parameter(torch.randn(self.responses).unsqueeze(
                dim=1).unsqueeze(dim=2))
        # Bias parameters start in zeros
        self.bias = nn.Parameter(torch.zeros(self.responses)) if bias1 else None
        # # # # # # # # # # # # # # The meshgrids
        # The orientations (NOT learnt!)
        self.thetas = nn.Parameter(torch.randn(self.total_kernels).unsqueeze(
            dim=1).unsqueeze(dim=2))
        # The original meshgrid
        # Bounding box
        xmin, xmax = -1, 1
        ymin, ymax = -1, 1
        x_space = torch.linspace(xmin, xmax, kernel_size)
        y_space = torch.linspace(ymin, ymax, kernel_size)
        (y, x) = torch.meshgrid(y_space, x_space)
        # Unsqueeze for all orientations
        x, y = x.unsqueeze(dim=0), y.unsqueeze(dim=0)
        # Add as parameters
        self.x = nn.Parameter(x, requires_grad=False)
        self.y = nn.Parameter(y, requires_grad=False)
        # Conv1x1 channels
        self.channels1x1 = self.responses * self.in_channels
        self.conv1x1 = nn.Conv2d(
            in_channels=self.channels1x1, out_channels=out_channels,
            kernel_size=1, bias=bias2)
        self.gabor_kernels = nn.Parameter(
            self.generate_gabor_kernels(), requires_grad=False)

    def forward(self, x):
        # Generate the Gabor kernels
        if self.training:
            self.gabor_kernels = nn.Parameter(
                self.generate_gabor_kernels(), requires_grad=False)
        # kernels are of shape
        # [self.kernels*self.orientations + self.extra_kernels, 1, self.kernel_size, self.kernel_size]
        # Reshape the input: x is of size
        # [batch_size, in_channels, H, W]
        # and we need to merge the batch size and the input channels for the
        # depthwise convolution (and include a '1' channel for the convolution)
        bs, _, H, W = x.size()
        x = x.view(bs*self.in_channels, H, W).unsqueeze(dim=1)
        # Perform convolution
        out = F.conv2d(input=x, weight=self.gabor_kernels, bias=self.bias,
            stride=self.stride, padding=self.padding)
        if self.relu:
            out = torch.relu(out)
        # 'out' is of size [batch_size*in_channels, newH, newW]
        _, _, newH, newW = out.size()
        # reshape to:
        out = out.view(bs, self.in_channels, self.responses, newH, newW)
        # now reshape to 
        out = out.view(bs, self.channels1x1, newH, newW)
        return self.conv1x1(out)

    '''
    Inspired by
    https://en.wikipedia.org/wiki/Gabor_filter
    '''
    def generate_gabor_kernels(self):
        sines = torch.sin(self.thetas)
        cosines = torch.cos(self.thetas)
        x = self.x * cosines - self.y * sines
        y = self.x * sines + self.y * cosines
        if self.use_translation:
            x = x + torch.tanh(self.trans_x)
            y = y + torch.tanh(self.trans_y)
        # Precompute some squared terms
        # ORIENTED KERNELS
        # Compute Gaussian term
        # gaussian_term = torch.exp(-.5 * ( (gamma_x**2 * x_t**2 + gamma_y**2 * y_t**2)/ sigma**2 ))
        ori_y_term = (self.gammas_y * y)**2
        exponent_ori = (x**2 + ori_y_term) * self.sigmas**2
        gaussian_term_ori = torch.exp(-exponent_ori)
        # Compute Cosine term
        # cosine_term = torch.cos(2 * np.pi * x_t / Lambda + psi)
        cosine_term_ori = torch.cos(x * self.Lambdas + self.psis)
        # 'ori_gb' has shape [self.kernels, self.orientations, kernel_size, kernel_size]
        ori_gb = gaussian_term_ori * cosine_term_ori
        if self.use_alphas:
            ori_gb = self.alphas * ori_gb
        return ori_gb.unsqueeze(dim=1)
