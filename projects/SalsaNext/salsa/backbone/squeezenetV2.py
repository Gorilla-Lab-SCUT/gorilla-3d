#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import torch
import torch.nn as nn
import torch.nn.functional as F


class Fire(nn.Module):
  def __init__(self,
               inplanes: int,
               squeeze_planes: int,
               expand1x1_planes: int,
               expand3x3_planes: int,
               bn_d: float=0.1):
    super(Fire, self).__init__()
    self.inplanes = inplanes
    self.bn_d = bn_d
    self.activation = nn.ReLU(inplace=True)
    self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
    self.squeeze_bn = nn.BatchNorm2d(squeeze_planes, momentum=self.bn_d)
    self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                               kernel_size=1)
    self.expand1x1_bn = nn.BatchNorm2d(expand1x1_planes, momentum=self.bn_d)
    self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                               kernel_size=3, padding=1)
    self.expand3x3_bn = nn.BatchNorm2d(expand3x3_planes, momentum=self.bn_d)

  def forward(self, x):
    x = self.activation(self.squeeze_bn(self.squeeze(x)))
    return torch.cat([
        self.activation(self.expand1x1_bn(self.expand1x1(x))),
        self.activation(self.expand3x3_bn(self.expand3x3(x)))
    ], 1)


class CAM(nn.Module):
  def __init__(self,
               inplanes: int,
               bn_d: float=0.1):
    super().__init__()
    self.inplanes = inplanes
    self.bn_d = bn_d
    self.pool = nn.MaxPool2d(7, 1, 3)
    self.squeeze = nn.Conv2d(inplanes, inplanes // 16,
                             kernel_size=1, stride=1)
    self.squeeze_bn = nn.BatchNorm2d(inplanes // 16, momentum=self.bn_d)
    self.relu = nn.ReLU(inplace=True)
    self.unsqueeze = nn.Conv2d(inplanes // 16, inplanes,
                               kernel_size=1, stride=1)
    self.unsqueeze_bn = nn.BatchNorm2d(inplanes, momentum=self.bn_d)
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    # 7x7 pooling
    y = self.pool(x)
    # squeezing and relu
    y = self.relu(self.squeeze_bn(self.squeeze(y)))
    # unsqueezing
    y = self.sigmoid(self.unsqueeze_bn(self.unsqueeze(y)))
    # attention
    return y * x

# ******************************************************************************

class SqueezeNetV2(nn.Module):
  """
     Class for Squeezeseg. Subclasses PyTorch's own "nn" module
  """

  def __init__(self,
               use_range: bool=True,
               use_xyz: bool=True,
               use_remission: bool=True,
               bn_d: float=0.01,
               dropout: float=0.01,
               output_stride: int=32,
               **kwargs):
    super().__init__()
    print("Using SqueezeNet Backbone")
    self.use_range = use_range
    self.use_xyz = use_xyz
    self.use_remission = use_remission
    self.bn_d = bn_d
    self.drop_prob = dropout
    self.output_stride = output_stride

    # input depth calc
    self.input_depth = 0
    self.input_idxs = []
    if self.use_range:
      self.input_depth += 1
      self.input_idxs.append(0)
    if self.use_xyz:
      self.input_depth += 3
      self.input_idxs.extend([1, 2, 3])
    if self.use_remission:
      self.input_depth += 1
      self.input_idxs.append(4)
    print("Depth of backbone input = ", self.input_depth)

    # stride play
    self.strides = [2, 2, 2, 2]
    # check current stride
    current_output_stride = 1
    for s in self.strides:
      current_output_stride *= s
    print("Original OS: ", current_output_stride)

    # make the new stride
    if self.output_stride > current_output_stride:
      print("Can't do OS, ", self.output_stride,
            " because it is bigger than original ", current_output_stride)
    else:
      # redo strides according to needed stride
      for i, stride in enumerate(reversed(self.strides), 0):
        if int(current_output_stride) != self.output_stride:
          if stride == 2:
            current_output_stride /= 2
            self.strides[-1 - i] = 1
          if int(current_output_stride) == self.output_stride:
            break
      print("New OS: ", int(current_output_stride))
      print("Strides: ", self.strides)

    # encoder
    self.conv1a = nn.Sequential(nn.Conv2d(self.input_depth, 64, kernel_size=3,
                                          stride=[1, self.strides[0]],
                                          padding=1),
                                nn.BatchNorm2d(64, momentum=self.bn_d),
                                nn.ReLU(inplace=True),
                                CAM(64, bn_d=self.bn_d))
    self.conv1b = nn.Sequential(nn.Conv2d(self.input_depth, 64, kernel_size=1,
                                          stride=1, padding=0),
                                nn.BatchNorm2d(64, momentum=self.bn_d))
    self.fire23 = nn.Sequential(nn.MaxPool2d(kernel_size=3,
                                             stride=[1, self.strides[1]],
                                             padding=1),
                                Fire(64, 16, 64, 64, bn_d=self.bn_d),
                                CAM(128, bn_d=self.bn_d),
                                Fire(128, 16, 64, 64, bn_d=self.bn_d),
                                CAM(128, bn_d=self.bn_d))
    self.fire45 = nn.Sequential(nn.MaxPool2d(kernel_size=3,
                                             stride=[1, self.strides[2]],
                                             padding=1),
                                Fire(128, 32, 128, 128, bn_d=self.bn_d),
                                Fire(256, 32, 128, 128, bn_d=self.bn_d))
    self.fire6789 = nn.Sequential(nn.MaxPool2d(kernel_size=3,
                                               stride=[1, self.strides[3]],
                                               padding=1),
                                  Fire(256, 48, 192, 192, bn_d=self.bn_d),
                                  Fire(384, 48, 192, 192, bn_d=self.bn_d),
                                  Fire(384, 64, 256, 256, bn_d=self.bn_d),
                                  Fire(512, 64, 256, 256, bn_d=self.bn_d))

    # output
    self.dropout = nn.Dropout2d(self.drop_prob)

    # last channels
    self.last_channels = 512

  def run_layer(self, x, layer, skips, output_stride):
    y = layer(x)
    if y.shape[2] < x.shape[2] or y.shape[3] < x.shape[3]:
      skips[output_stride] = x.detach()
      output_stride *= 2
    x = y
    return x, skips, output_stride

  def forward(self, x):
    # filter input
    x = x[:, self.input_idxs]

    # run cnn
    # store for skip connections
    skips = {}
    output_stride = 1

    # encoder
    skip_in = self.conv1b(x)
    x = self.conv1a(x)
    # first skip done manually
    skips[1] = skip_in.detach()
    output_stride *= 2

    x, skips, output_stride = self.run_layer(x, self.fire23, skips, output_stride)
    x, skips, output_stride = self.run_layer(x, self.dropout, skips, output_stride)
    x, skips, output_stride = self.run_layer(x, self.fire45, skips, output_stride)
    x, skips, output_stride = self.run_layer(x, self.dropout, skips, output_stride)
    x, skips, output_stride = self.run_layer(x, self.fire6789, skips, output_stride)
    x, skips, output_stride = self.run_layer(x, self.dropout, skips, output_stride)

    return x, skips

  def get_last_depth(self):
    return self.last_channels

  def get_input_depth(self):
    return self.input_depth

  