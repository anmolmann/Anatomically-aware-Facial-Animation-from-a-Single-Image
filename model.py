# model.py ---
#
# Filename: model.py

# Maintainer: Anmol Mann
# Description:
# Course Instructor: Kwang Moo Yi

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

# weights initilaization
# https://goo.gl/bqeW1K (CycleGAN and pix2pix)
# https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
# # Apply the weights_init_normal function to randomly initialize all weights
def weights_init_normal(input_tensor):
    classname = input_tensor.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch.nn.init.normal_(input_tensor.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1 or classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(input_tensor.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(input_tensor.bias.data, 0.0)

# The funciton is NOT actually used for training purpose.
def linearity_(flag = False, out_channel = 5):
    if flag == True:
        linear_ = nn.Sequential(
            nn.Linear(in_features=7056, out_features=4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=4096, out_features=2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=2048, out_features = out_channel),
            nn.Sigmoid()
        )
        return linear_
    else:
        non_linear_ = nn.Sequential(
            nn.Conv2d(3, 32, 9, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 16, 6, 1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 6, 1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 4, 2),
            nn.ReLU(),
            nn.Conv2d(16, 16, 4, 1),
            nn.ReLU()
        )
        return non_linear_

# https://hardikbansal.github.io/CycleGANBlog/
# a residue of input is added to the output.
class build_resnet(nn.Module):
    """
    Almost the same as implemented in Assignment 7's ResNet Model.
    """

    def __init__(self, in_channels, out_channels):
        super(build_resnet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias = False),
            nn.InstanceNorm2d(out_channels, affine = False),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias = False),
            nn.InstanceNorm2d(out_channels, affine = False))

        self.model.apply(weights_init_normal)

    def forward(self, x):
        x = x + self.model(x)
        return x

class Generate_GAN(build_resnet):
    """"
        Generator component of the model.
        Generates fake images
    """
    def __init__(self, out_channel = 64, in_channel = 5, num_reduc = 6):
        super(Generate_GAN, self).__init__(out_channel, out_channel)
        self.layers = []
        self.layers.append(nn.Conv2d(in_channel + 3, out_channel, 7, 1, 3, bias = False))
        self.layers.append(nn.InstanceNorm2d(out_channel, affine = False))
        self.layers.append(nn.ReLU(inplace = True)) # no copy made

        # Down-sampling done the same way as in PatchGANs (64 -> 128, 128 -> 256)
        num_layers = 2
        for index in range(0, num_layers):
            # 64, then 128
            in_channel = out_channel
            # 128, then 256
            out_channel = out_channel * 2
            self.layers.append(nn.Conv2d(in_channel, out_channel, 4, 2, 1, bias = False))
            self.layers.append(nn.InstanceNorm2d(out_channel, affine = False))
            self.layers.append(nn.ReLU(inplace = True))  # no copy made

        # Dimensionality reduction (like in Auto-encoders)
        for index in range(0, num_reduc):
            # https://pytorch.org/docs/master/nn.html?highlight=modulelist#torch.nn.Module.apply
            res_layer = build_resnet(out_channel, out_channel)
            self.layers.append(res_layer)

        # up-sampling
        for index in range(0, num_layers):
            in_channel = out_channel
            out_channel = out_channel // 2
            self.layers.append(nn.ConvTranspose2d(in_channel, out_channel, 4, 2, 1, bias = False))
            self.layers.append(nn.InstanceNorm2d(out_channel, affine = False))
            self.layers.append(nn.ReLU(inplace = True))

        # ouput layer
        self.model = nn.Sequential(*self.layers)

        # Color Mask (C)
        self.layers = []
        self.layers.append(nn.Conv2d(out_channel, 3, 7, 1, 3, bias = False))
        self.layers.append(nn.Tanh())
        self.color_mask = nn.Sequential(*self.layers)

        # Attention Mask (A)
        self.layers = []
        self.layers.append(nn.Conv2d(out_channel, 1, 7, 1, 3, bias = False))
        self.layers.append(nn.Sigmoid())  # Values between 0 and 1
        self.att_mask = nn.Sequential(*self.layers)

        self.model.apply(weights_init_normal)

    def forward(self, x, target):
        # giving RunTime Error: expected input[25, 20, 128, 128] to have 7 channels, but got 20 channels instead
        # target = target.view(target.size(0), target.size(1), 1, 1)
        # target = target.repeat(1, 1, x.size(2), x.size(3))
        target = target.unsqueeze(2).unsqueeze(3)
        target = target.expand(target.size(0), target.size(1), x.size(2), x.size(3))
        # Concatenate label embedding and image to produce input
        x = torch.cat([x, target], dim = 1)
        features = self.model(x)
        return self.att_mask(features), self.color_mask(features)

class Discriminate_GAN(nn.Module):
    """"
        Compare the real and the fake images.
        Tells how real the fake img actually is?
        Using PatchGAN
    """

    def __init__(self, img_size = 128, in_channel = 64, out_channel = 5, num_reduc = 6):
        super(Discriminate_GAN, self).__init__()
        self.out_channel = out_channel
        aus_out = 17
        self.layers = []
        self.layers.append(nn.Conv2d(3, in_channel, 4, 2, 1))
        # 0.2, True
        self.layers.append(nn.LeakyReLU(0.01))

        for index in range(1, num_reduc):
            self.layers.append(nn.Conv2d(in_channel, in_channel * 2, 4, 2, 1))
            self.layers.append(nn.LeakyReLU(0.01))
            in_channel = in_channel * 2

        self.model = nn.Sequential(*self.layers)
        self.real = nn.Conv2d(in_channel, 1, 3, 1, 1, bias = False)
        self.fake = []
        self.fake.append(nn.Conv2d(in_channel, aus_out, int(img_size / np.power(2, num_reduc)), bias = False))
        self.fake.append(nn.Sigmoid())
        self.fake= nn.Sequential(*self.fake)

        self.model.apply(weights_init_normal)

    def forward(self, x):
        real_img = self.real(self.model(x))

        """
        non_linear = linearity_(False, self.out_channel)
        non_linear = non_linear(x).view(x.size()[0], -1)
        linear = linearity_(True, self.out_channel)
        linear = linear(non_linear)
        """

        fake_img = self.fake(self.model(x))
        return real_img.squeeze(), fake_img.squeeze()
        # This line was giving error so had to swtich to squeeze()
        # return real_img.squeeze(), fake_img.view(fake_img.size(0), fake_img.size(1))
