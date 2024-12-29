from __future__ import annotations
from typing import Optional, Dict
import torch
from torch import Tensor
import torch.nn as nn


def gram_schmidt(input):
    def projection(u, v):
        return (v * u).sum() / (u * u).sum() * u
    output = []
    for x in input:
        for y in output:
            x = x - projection(y, x)
        x = x/x.norm(p=2)
        output.append(x)
    return torch.stack(output)

def initialize_orthogonal_filters(c, h, w):

    if h*w < c:
        n = c//(h*w)
        gram = []
        for i in range(n):
            gram.append(gram_schmidt(torch.rand([h * w, 1, h, w])))
        return torch.cat(gram, dim=0)
    else:
        return gram_schmidt(torch.rand([c, 1, h, w]))
class GramSchmidtTransform(torch.nn.Module):
    instance: Dict[int, Optional[GramSchmidtTransform]] = {}
    constant_filter: Tensor

    @staticmethod
    def build(c: int, h: int):
        if c not in GramSchmidtTransform.instance:
            GramSchmidtTransform.instance[(c, h)] = GramSchmidtTransform(c, h)
        return GramSchmidtTransform.instance[(c, h)]

    def __init__(self, c: int, h: int):
        super().__init__()
        with torch.no_grad():
            rand_ortho_filters = initialize_orthogonal_filters(c, h, h).view(c, h, h)
        self.register_buffer("constant_filter", rand_ortho_filters.detach())

    def forward(self, x):
        _, _, h, w = x.shape
        _, H, W = self.constant_filter.shape
        if h != H or w != W: x = torch.nn.functional.adaptive_avg_pool2d(x, (H, W))
        return (self.constant_filter * x).sum(dim=(-1, -2), keepdim=True)
class Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, FWT: GramSchmidtTransform, input: Tensor):
        while input[0].size(-1) > 1:
            input = FWT(input)
        b = input.size(0)
        return input.view(b, -1)

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class OCA1(nn.Module):
    def __init__(self, inplanes,planes, height, stride=1, downsample=None):
        super(OCA1, self).__init__()
        self._process: nn.Module = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes * 4),
        )
        self.conv1x1 = nn.Conv2d(planes*4,inplanes,1)
        self.downsample = downsample
        self.stride = stride

        self.planes = planes
        self._excitation = nn.Sequential(
            nn.Linear(in_features=4 * planes, out_features=round(planes / 4), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=round(planes / 4), out_features=4 * planes, bias=False),
            nn.Sigmoid(),
        )
        self.OrthoAttention =Attention()
        self.F_C_A = GramSchmidtTransform.build(4 * planes, height)
    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self._process(x)
        compressed = self.OrthoAttention(self.F_C_A, out)
        b, c = out.size(0), out.size(1)
        attention = self._excitation(compressed).view(b, c, 1, 1)
        attention = attention * out
        attention = self.conv1x1(attention)
        attention += residual
        activated = torch.relu(attention)
        return activated

class OCA2(nn.Module):
    def __init__(self, inplanes, planes, height, stride=1, downsample=None):
        super(OCA2, self).__init__()

        self._preprocess: nn.Module = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
        )
        self._scale: nn.Module = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes * 4),
        )
        self.conv1x1 = nn.Conv2d(planes * 4, inplanes, 1)
        self.downsample = downsample
        self.stride = stride
        self.planes = planes

        self._excitation = nn.Sequential(
            nn.Linear(in_features=planes, out_features=round(planes / 16), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=round(planes / 16), out_features=planes, bias=False),
            nn.Sigmoid(),
        )
        self.OrthoAttention = Attention()
        self.F_C_A = GramSchmidtTransform.build(planes, height)

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        preprocess_out = self._preprocess(x)
        compressed = self.OrthoAttention(self.F_C_A, preprocess_out)
        b, c = preprocess_out.size(0), preprocess_out.size(1)
        attention = self._excitation(compressed).view(b, c, 1, 1)
        attentended = attention * preprocess_out
        scale_out = self._scale(attentended)
        scale_out = self.conv1x1(scale_out)
        scale_out += residual
        activated = torch.relu(scale_out)
        return activated

if __name__ == '__main__':
    input = torch.randn(32, 512, 8, 8)
    block = OCA2(inplanes=512,planes=512, height=256)
    output = block(input)
    print(f"Input shape: {input.shape}")
    print(f"Output shape: {output.shape}")