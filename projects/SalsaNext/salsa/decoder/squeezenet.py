# Adapted from https://github.com/BichenWuUCB/SqueezeSeg
from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FireUp(nn.Module):

  def __init__(self,
               inplanes: int,
               squeeze_planes: int,
               expand1x1_planes: int,
               expand3x3_planes: int,
               stride: int):
    super().__init__()
    self.inplanes = inplanes
    self.stride = stride
    self.activation = nn.ReLU(inplace=True)
    self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
    if self.stride == 2:
      self.upconv = nn.ConvTranspose2d(squeeze_planes, squeeze_planes,
                                       kernel_size=[1, 4], stride=[1, 2],
                                       padding=[0, 1])
    self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                               kernel_size=1)
    self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                               kernel_size=3, padding=1)

  def forward(self, x):
    x = self.activation(self.squeeze(x))
    if self.stride == 2:
      x = self.activation(self.upconv(x))
    return torch.cat([
        self.activation(self.expand1x1(x)),
        self.activation(self.expand3x3(x))
    ], 1)


# ******************************************************************************

class SqueezeDecoder(nn.Module):
  """
     Class for DarknetSeg. Subclasses PyTorch's own "nn" module
  """

  def __init__(self,
               output_stride=32,
               feature_depth=512,
               dropout: float=0.01,
               **kwargs):
    super(SqueezeDecoder, self).__init__()
    self.backbone_output_stride = output_stride
    self.backbone_feature_depth = feature_depth
    self.drop_prob = dropout

    # stride play
    self.strides = [2, 2, 2, 2]
    # check current stride
    current_output_stride = np.product(self.strides)
    print("Decoder original output_stride: ", current_output_stride)
    # redo strides according to needed stride
    for i, stride in enumerate(self.strides):
      if int(current_output_stride) != self.backbone_output_stride:
        if stride == 2: 
          current_output_stride /= 2
          self.strides[i] = 1
        if int(current_output_stride) == self.backbone_output_stride:
          break
    print("Decoder new output_stride: ", int(current_output_stride))
    print("Decoder strides: ", self.strides)

    # decoder
    # decoder
    self.firedec10 = FireUp(self.backbone_feature_depth, 64, 128, 128,
                            stride=self.strides[0])
    self.firedec11 = FireUp(256, 32, 64, 64,
                            stride=self.strides[1])
    self.firedec12 = FireUp(128, 16, 32, 32,
                            stride=self.strides[2])
    self.firedec13 = FireUp(64, 16, 32, 32,
                            stride=self.strides[3])

    # layer list to execute with skips
    self.layers = [self.firedec10, self.firedec11,
                   self.firedec12, self.firedec13]

    # for a bit of fun
    self.dropout = nn.Dropout2d(self.drop_prob)

    # last channels
    self.last_channels = 64

  def run_layer(self, x, layer, skips, output_stride):
    feats = layer(x)  # up
    if feats.shape[-1] > x.shape[-1]:
      output_stride //= 2  # match skip
      feats = feats + skips[output_stride].detach()  # add skip
    x = feats
    return x, skips, output_stride

  def forward(self, x, skips):
    output_stride = self.backbone_output_stride

    # run layers
    x, skips, output_stride = self.run_layer(x, self.firedec10, skips, output_stride)
    x, skips, output_stride = self.run_layer(x, self.firedec11, skips, output_stride)
    x, skips, output_stride = self.run_layer(x, self.firedec12, skips, output_stride)
    x, skips, output_stride = self.run_layer(x, self.firedec13, skips, output_stride)

    x = self.dropout(x)

    return x

  def get_last_depth(self):
    return self.last_channels