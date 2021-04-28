# This file was modified from https://github.com/BobLiu20/YOLOv3_PyTorch
# It needed to be modified in order to accomodate for different strides in the

import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F


class BasicBlock(nn.Module):
  def __init__(self, inplanes, planes, bn_d=0.1):
    super(BasicBlock, self).__init__()
    self.conv1 = nn.Conv2d(inplanes, planes[0], kernel_size=1,
                           stride=1, padding=0, bias=False)
    self.bn1 = nn.BatchNorm2d(planes[0], momentum=bn_d)
    self.relu1 = nn.LeakyReLU(0.1)
    self.conv2 = nn.Conv2d(planes[0], planes[1], kernel_size=3,
                           stride=1, padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(planes[1], momentum=bn_d)
    self.relu2 = nn.LeakyReLU(0.1)

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu1(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu2(out)

    out += residual
    return out


# ******************************************************************************

# number of layers per model
model_blocks = {
    21: [1, 1, 2, 2, 1],
    53: [1, 2, 8, 8, 4],
}


class DarkNet(nn.Module):
  """
     Class for DarknetSeg. Subclasses PyTorch's own "nn" module
  """

  # def __init__(self, params):
  def __init__(self,
               use_range: bool=True,
               use_xyz: bool=True,
               use_remission: bool=True,
               dropout: float=0.01,
               bn_d: float=0.1,
               output_stride: int=32,
               layers: int=53,
               **kwargs):
    super().__init__()
    self.use_range = use_range
    self.use_xyz = use_xyz
    self.use_remission = use_remission
    self.drop_prob = dropout
    self.bn_d = bn_d
    self.output_stride = output_stride
    self.layers = layers
    print("Using DarknetNet" + str(self.layers) + " Backbone")

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
    self.strides = [2, 2, 2, 2, 2]
    # check current stride
    current_output_stride = 1
    for s in self.strides:
      current_output_stride *= s
    print("Original output_stride: ", current_output_stride)

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
      print("New output_stride: ", int(current_output_stride))
      print("Strides: ", self.strides)

    # check that darknet exists
    assert self.layers in model_blocks.keys()

    # generate layers depending on darknet type
    self.blocks = model_blocks[self.layers]

    # input layer
    self.conv1 = nn.Conv2d(self.input_depth, 32, kernel_size=3,
                           stride=1, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(32, momentum=self.bn_d)
    self.relu1 = nn.LeakyReLU(0.1)

    # encoder
    self.enc1 = self._make_enc_layer(BasicBlock, [32, 64], self.blocks[0],
                                     stride=self.strides[0], bn_d=self.bn_d)
    self.enc2 = self._make_enc_layer(BasicBlock, [64, 128], self.blocks[1],
                                     stride=self.strides[1], bn_d=self.bn_d)
    self.enc3 = self._make_enc_layer(BasicBlock, [128, 256], self.blocks[2],
                                     stride=self.strides[2], bn_d=self.bn_d)
    self.enc4 = self._make_enc_layer(BasicBlock, [256, 512], self.blocks[3],
                                     stride=self.strides[3], bn_d=self.bn_d)
    self.enc5 = self._make_enc_layer(BasicBlock, [512, 1024], self.blocks[4],
                                     stride=self.strides[4], bn_d=self.bn_d)

    # for a bit of fun
    self.dropout = nn.Dropout2d(self.drop_prob)

    # last channels
    self.last_channels = 1024

  # make layer useful function
  def _make_enc_layer(self, block, planes, blocks, stride, bn_d=0.1):
    layers = []

    #  downsample
    layers.append(("conv", nn.Conv2d(planes[0], planes[1],
                                     kernel_size=3,
                                     stride=[1, stride], dilation=1,
                                     padding=1, bias=False)))
    layers.append(("bn", nn.BatchNorm2d(planes[1], momentum=bn_d)))
    layers.append(("relu", nn.LeakyReLU(0.1)))

    #  blocks
    inplanes = planes[1]
    for i in range(0, blocks):
      layers.append(("residual_{}".format(i),
                     block(inplanes, planes, bn_d)))

    return nn.Sequential(OrderedDict(layers))

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

    # first layer
    x, skips, output_stride = self.run_layer(x, self.conv1, skips, output_stride)
    x, skips, output_stride = self.run_layer(x, self.bn1, skips, output_stride)
    x, skips, output_stride = self.run_layer(x, self.relu1, skips, output_stride)

    # all encoder blocks with intermediate dropouts
    x, skips, output_stride = self.run_layer(x, self.enc1, skips, output_stride)
    x, skips, output_stride = self.run_layer(x, self.dropout, skips, output_stride)
    x, skips, output_stride = self.run_layer(x, self.enc2, skips, output_stride)
    x, skips, output_stride = self.run_layer(x, self.dropout, skips, output_stride)
    x, skips, output_stride = self.run_layer(x, self.enc3, skips, output_stride)
    x, skips, output_stride = self.run_layer(x, self.dropout, skips, output_stride)
    x, skips, output_stride = self.run_layer(x, self.enc4, skips, output_stride)
    x, skips, output_stride = self.run_layer(x, self.dropout, skips, output_stride)
    x, skips, output_stride = self.run_layer(x, self.enc5, skips, output_stride)
    x, skips, output_stride = self.run_layer(x, self.dropout, skips, output_stride)

    return x, skips

  def get_last_depth(self):
    return self.last_channels

  def get_input_depth(self):
    return self.input_depth

