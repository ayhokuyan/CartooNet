import torch.nn as nn
import torch
from torch.nn.utils import spectral_norm
from networks.commons import Mean


def calc(pad, h, k, s):
  import math
  return math.floor((h + 2 * pad - (k - 1) - 1) / s + 1)
  
class Conv2DNormLReLU(nn.Module):
  def __init__(self,
               in_channels: int,
               out_channels: int,
               kernel_size: int = 3,
               stride: int = 1,
               padding: int = 1,
               bias=False) -> None:
    super().__init__()
    self.conv = nn.Conv2d(in_channels, out_channels,
                          kernel_size, stride,
                          padding, bias=bias)
    # NOTE layer norm is crucial for animegan!
    self.norm = nn.GroupNorm(1, out_channels,
                             affine=True)
    self.lrelu = nn.LeakyReLU(0.2, inplace=True)

  def forward(self, x):
    x = self.conv(x)
    x = self.norm(x)
    x = self.lrelu(x)
    return x
  
class InvertedresBlock(nn.Module):
  def __init__(self, in_channels: int,
               expansion: float,
               out_channels: int,
               bias=False):
    super().__init__()
    self.in_channels = in_channels
    self.expansion = expansion
    self.out_channels = out_channels
    self.bottle_channels = round(self.expansion * self.in_channels)
    self.body = nn.Sequential(
        # pw
        Conv2DNormLReLU(self.in_channels, self.bottle_channels, kernel_size=1, bias=bias),
        # dw
        nn.Conv2d(self.bottle_channels, self.bottle_channels, kernel_size=3,
                  stride=1, padding=0, groups=self.bottle_channels, bias=True),
        nn.GroupNorm(1, self.bottle_channels, affine=True),
        nn.LeakyReLU(0.2, inplace=True),
        # pw & linear
        nn.Conv2d(self.bottle_channels, self.out_channels, kernel_size=1, padding=0, bias=False),
        nn.GroupNorm(1, self.out_channels, affine=True),
    )
    
  def forward(self, x0):
    x = self.body(x0)
    if self.in_channels == self.out_channels:
      out = torch.add(x0, x)
    else:
      out = x
    return x


class UnetGeneratorIRB(nn.Module):
  def __init__(self, channel=32, num_blocks=4):
    super().__init__()

    self.conv = nn.Conv2d(3, channel, [7, 7], padding=3)  # same 256,256
    self.conv1 = nn.Conv2d(channel, channel, [3, 3], stride=2, padding=1)  # same 128,128
    self.conv2 = nn.Conv2d(channel, channel * 2, [3, 3], padding=1)  # 128,128
    self.conv3 = nn.Conv2d(channel * 2, channel * 2, [3, 3], stride=2, padding=1)  # 64,64
    self.conv4 = nn.Conv2d(channel * 2, channel * 4, [3, 3], padding=1)  # 64,64

    self.mid = nn.Sequential(
        InvertedresBlock(channel * 4, 2, channel * 8),
        InvertedresBlock(channel * 8, 2, channel * 8),
        InvertedresBlock(channel * 8, 2, channel * 8),
        InvertedresBlock(channel * 8, 2, channel * 8),
        Conv2DNormLReLU(channel * 8, channel * 4))

    self.conv5 = nn.Conv2d(channel * 4, channel * 2, [3, 3], padding=1)  # 64,64
    self.conv6 = nn.Conv2d(channel * 2, channel * 2, [3, 3], padding=1)  # 64,64
    self.conv7 = nn.Conv2d(channel * 2, channel, [3, 3], padding=1)  # 64,64
    self.conv8 = nn.Conv2d(channel, channel, [3, 3], padding=1)  # 64,64
    self.conv9 = nn.Conv2d(channel, 3, [7, 7], padding=3)  # 64,64

    self.leak_relu = nn.LeakyReLU(inplace=True)
    self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
    self.act = nn.Tanh()

  def forward(self, inputs):
    x0 = self.conv(inputs)
    x0 = self.leak_relu(x0)  # 256, 256, 32

    x1 = self.conv1(x0)
    x1 = self.leak_relu(x1)
    x1 = self.conv2(x1)
    x1 = self.leak_relu(x1)  # 128, 128, 64

    x2 = self.conv3(x1)
    x2 = self.leak_relu(x2)
    x2 = self.conv4(x2)
    x2 = self.leak_relu(x2)  # 64, 64, 128

    #x2 = self.resblock(x2)  # 64, 64, 128
    x2 = self.mid(x2)
    
    x2 = self.conv5(x2)
    x2 = self.leak_relu(x2)  # 64, 64, 64
    

    x3 = self.upsample(x2)
    x3 = self.conv6(x3 + x1)
    x3 = self.leak_relu(x3)
    x3 = self.conv7(x3)
    x3 = self.leak_relu(x3)  # 128, 128, 32

    x4 = self.upsample(x3)
    x4 = self.conv8(x4 + x0)
    x4 = self.leak_relu(x4)
    x4 = self.conv9(x4)  # 256, 256, 32

    return self.act(x4)


class SpectNormDiscriminator(nn.Module):
  def __init__(self, channel=32, patch=True):
    super().__init__()
    self.channel = channel
    self.patch = patch
    in_channel = 3
    l = []
    for idx in range(3):
      l.extend([
          spectral_norm(nn.Conv2d(in_channel, channel * (2**idx), 3, stride=2, padding=1)),
          nn.LeakyReLU(inplace=True),
          spectral_norm(nn.Conv2d(channel * (2**idx), channel * (2**idx), 3, stride=1, padding=1)),
          nn.LeakyReLU(inplace=True),
      ])
      in_channel = channel * (2**idx)
    self.body = nn.Sequential(*l)
    if self.patch:
      self.head = spectral_norm(nn.Conv2d(in_channel, 1, 1, padding=0))
    else:
      self.head = nn.Sequential(Mean([1, 2]), nn.Linear(in_channel, 1))

  def forward(self, x):
    x = self.body(x)
    x = self.head(x)
    return x
