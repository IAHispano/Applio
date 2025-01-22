import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm

from rvc.lib.algorithm.pqmf import PQMF

class MBD1(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv1d(   1,   16, kernel_size= 7, stride=1, padding=3)),
            weight_norm(nn.Conv1d(  16,   64, kernel_size=11, stride=1, padding=5, groups=4)),
            weight_norm(nn.Conv1d(  64,  256, kernel_size=11, stride=4, padding=5, groups=16)),
            weight_norm(nn.Conv1d( 256, 1024, kernel_size=11, stride=4, padding=5, groups=64)),
            weight_norm(nn.Conv1d(1024, 1024, kernel_size=11, stride=4, padding=5, groups=256)),
            weight_norm(nn.Conv1d(1024, 1024, kernel_size= 5, stride=1, padding=2)),
        ])
        self.conv_post = weight_norm(nn.Conv1d(1024, 1, kernel_size=3, stride=1))
        
    def forward(self, x):
        fmap = []
        for block in self.convs:
            x = block(x)
            x = F.leaky_relu(x, 0.2, inplace=True)
            fmap.append(x)
        x = self.conv_post(x)
        return x, fmap

class MBD2(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv1d(   1,   16, kernel_size=11, stride=1, padding=5)),
            weight_norm(nn.Conv1d(  16,   64, kernel_size=21, stride=1, padding=10, groups=4)),
            weight_norm(nn.Conv1d(  64,  256, kernel_size=21, stride=4, padding=10, groups=16)),
            weight_norm(nn.Conv1d( 256, 1024, kernel_size=21, stride=4, padding=10, groups=64)),
            weight_norm(nn.Conv1d(1024, 1024, kernel_size=21, stride=4, padding=10, groups=256)),
            weight_norm(nn.Conv1d(1024, 1024, kernel_size= 5, stride=1, padding=2)),
        ])
        self.conv_post = weight_norm(nn.Conv1d(1024, 1, kernel_size=3, stride=1))
        
    def forward(self, x):
        fmap = []
        for block in self.convs:
            x = block(x)
            x = F.leaky_relu(x, 0.2, inplace=True)
            fmap.append(x)
        x = self.conv_post(x)
        return x, fmap
        
class MBD3(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv1d(   1,   16, kernel_size=15, stride=1, padding=7)),
            weight_norm(nn.Conv1d(  16,   64, kernel_size=41, stride=1, padding=20, groups=4)),
            weight_norm(nn.Conv1d(  64,  256, kernel_size=41, stride=4, padding=20, groups=16)),
            weight_norm(nn.Conv1d( 256, 1024, kernel_size=41, stride=4, padding=20, groups=64)),
            weight_norm(nn.Conv1d(1024, 1024, kernel_size=41, stride=4, padding=20, groups=256)),
            weight_norm(nn.Conv1d(1024, 1024, kernel_size= 5, stride=1, padding=2)),
        ])
        self.conv_post = weight_norm(nn.Conv1d(1024, 1, kernel_size=3, stride=1))
        
    def forward(self, x):
        fmap = []
        for block in self.convs:
            x = block(x)
            x = F.leaky_relu(x, 0.2, inplace=True)
            fmap.append(x)
        x = self.conv_post(x)
        return x, fmap
        
class CoMBD(torch.nn.Module):
    def __init__(self):
        super().__init__()
               
        self.pqmf1 = PQMF(4, 192, 0.13, 10.0)
        self.pqmf2 = PQMF(2, 256, 0.25, 10.0)

        self.mbd1 = MBD1()
        self.mbd2 = MBD2()
        self.mbd3 = MBD3()

    def forward(self, y, y_hat):
        outs_real, f_maps_real, outs_fake, f_maps_fake = [], [], [], []
        
        y1, y2 = self.pqmf1.analysis(y)[:, :1, :], self.pqmf2.analysis(y)[:, :1, :]
        yh1, yh2 = self.pqmf1.analysis(y_hat)[:, :1, :], self.pqmf2.analysis(y_hat)[:, :1, :]
        
        outs_r, maps_r = self.mbd1(y1)
        outs_g, maps_g = self.mbd1(yh1)
        
        outs_real.append(outs_r)
        f_maps_real.append(maps_r)
        outs_fake.append(outs_g)
        f_maps_fake.append(maps_g)
        
        outs_r, maps_r = self.mbd2(y2)
        outs_g, maps_g = self.mbd2(yh2)
        
        outs_real.append(outs_r)
        f_maps_real.append(maps_r)
        outs_fake.append(outs_g)
        f_maps_fake.append(maps_g)
        
        outs_r, maps_r = self.mbd3(y)
        outs_g, maps_g = self.mbd3(y_hat)
        
        outs_real.append(outs_r)
        f_maps_real.append(maps_r)
        outs_fake.append(outs_g)
        f_maps_fake.append(maps_g)

        return outs_real, outs_fake, f_maps_real, f_maps_fake