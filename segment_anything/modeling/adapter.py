import torch
from torch import nn
#from .common import LayerNorm2d
from torch.nn import functional as F

class ChannelAttention(nn.Module):  #N,C,H,W
    def __init__(self, channels=32, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channels, channels // reduction, bias=False),nn.GELU(),nn.Linear(channels // reduction, channels, bias=False),nn.Sigmoid())
    def forward(self, x):
        N, C, _, _ = x.shape
        x1 = self.avg_pool(x) + self.max_pool(x) #N,C,1,1
        x1 = x1.view(N, C)
        x1 = self.fc(x1)
        x1 = x1.view(N, C, 1, 1)
        return x * x1           #N,C,H,W


class MutiScaleAdapter(nn.Module):
    def __init__(self, in_dim=768, hidden_dim=192, dilation_rates=[1, 3, 5]):
        super().__init__()
        self.down_project = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.up_project = nn.Linear(hidden_dim, in_dim)
        self.dilation_rates = dilation_rates
        self.base_weight = nn.Parameter(torch.randn(hidden_dim//3, hidden_dim//6, 3, 3))
        self.delta_weights = nn.ParameterList([nn.Parameter(torch.zeros(hidden_dim//3, hidden_dim//6, 3, 3)) for i in range(len(dilation_rates)-1)])
        self.channel_attention = nn.ModuleList([ChannelAttention(channels=hidden_dim//3) for _ in range(len(dilation_rates))])
        self.fuse_conv = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, 1), nn.GELU())
        self.convlist = nn.ModuleList([nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim//6, 1), nn.GELU()) for _ in range(len(dilation_rates))])
    def forward(self, x):    #x:N,H,W,768
        branch_outputs = []
        down_x = self.down_project(x)
        down_x = self.act(down_x)
        down_x = down_x.permute(0, 3, 1, 2).contiguous()
        for i, rate in enumerate(self.dilation_rates):
            if i == 0:
                weight = self.base_weight
            else:
                weight = self.base_weight + self.delta_weights[i-1]
            padding = rate * (3 - 1) // 2
            down_x1 = self.convlist[i](down_x)
            out = F.conv2d(down_x1,weight=weight,bias=None,padding=padding,dilation=rate)
            out = self.act(out)
            out = self.channel_attention[i](out)
            branch_outputs.append(out)
        output = torch.cat(branch_outputs, 1)
        output = self.fuse_conv(output)        #N,32,H,W
        output = output.permute(0, 2, 3, 1).contiguous() #N,H,W,32
        up_x = self.up_project(output)   #N,H,W,768
        return up_x + x

