from abc import ABC

import torch
from torch import nn
from torch.nn import functional as F, init


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        output = self.sigmoid(max_out + avg_out)
        return output

""" Spatial Attention Module"""

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()
        # self.conv1 = nn.Conv2d(256, 1,1,1)

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        # conv_result = self.conv1(x)
        # output1 = self.sigmoid(conv_result)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output

class CSAMBlock(nn.Module):

    def __init__(self, channel=512, reduction=16, kernel_size=49):
        super().__init__()
        self.ca = ChannelAttention(channel=channel,reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        residual = x
        out = x*self.ca(x)
        out = out*self.sa(out)
        return out+residual

class CSAMBlock432(nn.Module):

    def __init__(self, channel=512, reduction=16, kernel_size=49):
        super().__init__()
        self.ca = ChannelAttention(channel=channel,reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)
        self.convl = nn.Conv2d(channel * 2, channel, kernel_size=1, stride=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x, lca_down):
        b, c, _, _ = x.size()
        residual = x
        out = x*self.ca(x)
        out = out*self.sa(out)
        lca_down = self.convl(lca_down)
        lca_down = self.upsample(lca_down)
        return out+residual+lca_down


class CSAMBlock1(nn.Module):

    def __init__(self, channel=512, reduction=16, kernel_size=49):
        super().__init__()
        self.ca = ChannelAttention(channel=channel,reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)
        # self.convl = nn.Conv2d(channel * 2, channel, kernel_size=1, stride=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x, lca_down):
        b, c, _, _ = x.size()
        residual = x
        out = x*self.ca(x)
        out = out*self.sa(out)
        lca_down = self.upsample(lca_down)
        return out+residual+lca_down


""" Local Context Attention Module"""

# 局部上下文注意： 关注分值不确定的区域，更加关注预测图越靠近0.5的区域，减少趋近0和1的关注-----------------------------4
class LCA44(nn.Module):
    def __init__(self, channel, reduction=16, kernel_size=7):
        super(LCA44, self).__init__()
        self.ca = ChannelAttention(channel=channel, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x, pred):
        residual = x
        x = x * self.ca(x)
        score = torch.sigmoid(pred)
        dist = torch.abs(score - 0.5)
        att = 1 - (dist / 0.5)

        att_x = x * att
        y = att_x * self.sa(att_x)
        out = y + residual

        return out

# 局部上下文注意： 关注分值不确定的区域，更加关注预测图越靠近0.5的区域，减少趋近0和1的关注-----------------------------3 2
class LCA32(nn.Module):
    def __init__(self, in_channels, channel, reduction=16, kernel_size=7):
        super(LCA32, self).__init__()
        self.convl = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, stride=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.ca = ChannelAttention(channel=channel, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)


    def forward(self, x, pred, lca_down):
        residual = x
        x = x * self.ca(x)
        score = torch.sigmoid(pred)
        dist = torch.abs(score - 0.5)
        att = 1 - (dist / 0.5)

        att_x = x * att
        lca_down = self.convl(lca_down)
        lca_down = self.upsample(lca_down)
        y = att_x + lca_down
        y = y * self.sa(y)
        out = y + residual

        return out


# 局部上下文注意： 关注分值不确定的区域，更加关注预测图越靠近0.5的区域，减少趋近0和1的关注----------------------------- 1
class LCA11(nn.Module):
    def __init__(self, channel, reduction=16, kernel_size=7):
        super(LCA11, self).__init__()
        # self.convl = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, stride=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.ca = ChannelAttention(channel=channel, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x, pred, lca_down):
        residual = x
        x = x * self.ca(x)
        score = torch.sigmoid(pred)
        dist = torch.abs(score - 0.5)
        att = 1 - (dist / 0.5)

        att_x = x * att
        lca_down = self.upsample(lca_down)
        y = att_x + lca_down
        y = y * self.sa(y)
        out = y + residual

        return out


class RA(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RA, self).__init__()
        self.convert = nn.Conv2d(in_channel, out_channel, 1)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(True)
        self.convs = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, padding=1), nn.BatchNorm2d(out_channel), nn.ReLU(True),
            nn.Conv2d(out_channel, out_channel, 3, padding=1), nn.BatchNorm2d(out_channel), nn.ReLU(True),
            nn.Conv2d(out_channel, out_channel, 3, padding=1), nn.BatchNorm2d(out_channel), nn.ReLU(True),
            nn.Conv2d(out_channel, 1, 3, padding=1),
        )
        self.channel = out_channel

    def forward(self, x, y):    # x4(32, 1024, 22, 22), y5_4(32, 1, 22, 22)
        a = torch.sigmoid(-y)
        x = self.relu(self.bn(self.convert(x)))
        x = a.expand(-1, self.channel, -1, -1).mul(x)
        y = y + self.convs(x)

        return y

    def initialize(self):
        weight_init(self)




def weight_init(module):
    for n, m in module.named_children():
        print('initialize: '+n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, nn.ReLU):
            pass
        else:
            m.initialize()



""" Reverse Attention Module"""

class RA1(nn.Module):
    def __init__(self, in_channel, reduction=16, kernel_size=7):
        super(RA1, self).__init__()
        self.channel = in_channel
        # self.convl = nn.Conv2d(in_channel_lca, in_channel_lca // 2, kernel_size=1, stride=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

        self.ca = ChannelAttention(channel=in_channel,reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)

        self.selayer = SELayer(in_channel)

    def forward(self, x, pred, lca):
        residual = x
        x = x * self.ca(x)
        # x = x * self.sa(x)
        a = torch.sigmoid(-pred)
        # pred = torch.sigmoid(pred)
        # a = 1 - pred
        att = a.expand(-1, self.channel, -1, -1).mul(x)       # a与x逐元素乘法
        lca = self.upsample(lca)
        # att1 = att * self.ca(att)
        att = att * self.sa(att)
        # att = torch.cat([att1, att2], dim=1)
        # att = att1 + att2

        y = att + residual + lca

        return y


""" Local Context Attention Module"""

class RA23(nn.Module):
    def __init__(self, in_channel, in_channel_lca, reduction=16, kernel_size=7):
        super(RA23, self).__init__()
        self.channel = in_channel
        self.convl = nn.Conv2d(in_channel_lca, in_channel_lca // 2, kernel_size=1, stride=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

        self.ca = ChannelAttention(channel=in_channel,reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)
        self.selayer = SELayer(in_channel)

    def forward(self, x, pred, lca):
        residual = x
        x = x * self.ca(x)
        # x = x * self.sa(x)
        a = torch.sigmoid(-pred)
        # x = self.relu(self.bn(self.convert(x)))
        att = a.expand(-1, self.channel, -1, -1).mul(x)       # a与x逐元素乘法
        att = att * self.sa(att)

        lca = self.convl(lca)
        lca = self.upsample(lca)

        y = att + residual + lca
        return y


""" Local Context Attention Module"""

class RA4(nn.Module):
    def __init__(self, in_channel, reduction=16, kernel_size=7):
        super(RA4, self).__init__()
        self.channel = in_channel

        self.ca = ChannelAttention(channel=in_channel,reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x, pred):
        residual = x
        x = x * self.ca(x)
        # x = x * self.sa(x)

        a = torch.sigmoid(-pred)
        # x = self.relu(self.bn(self.convert(x)))
        att = a.expand(-1, self.channel, -1, -1).mul(x)       # a与x逐元素乘法
        att = att * self.sa(att)

        y = att + residual
        return y




""" Modify """
""" Global Context Module_RFB"""

class GCM_RFB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCM_RFB, self).__init__()
        out_channel_list = [256, 128, 64, 64]
        upsampe_scale = [2, 4, 8, 16]
        GClist = []
        GCoutlist = []

        GClist.append(nn.Sequential(    # Conv1x1 -> Conv3x3, rate=1
            nn.Conv2d(in_channels, out_channels, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, dilation=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        ))
        # GClist.append(nn.Sequential(    # Conv3x3 -> Conv3x3, rate=3
        #     nn.Conv2d(in_channels, out_channels, 3, padding=1, dilation=1),
        #     nn.BatchNorm2d(out_channels),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(out_channels, out_channels, 3, padding=2, dilation=2),
        #     nn.BatchNorm2d(out_channels),
        #     nn.ReLU(True),
        # ))
        GClist.append(nn.Sequential(    # Conv3x3 -> Conv3x3, rate=3
            nn.Conv2d(in_channels, out_channels, 3, padding=1, dilation=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=3, dilation=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        ))
        GClist.append(nn.Sequential(    # Conv5x5 -> Conv3x3, rate=5
            nn.Conv2d(in_channels, out_channels, 5, padding=1, dilation=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=5, dilation=5),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        ))
        GClist.append(nn.Sequential(            # Non-Local
            nn.Conv2d(in_channels, out_channels, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # NonLocalBlock(out_channels)
        ))
        self.GCmodule = nn.ModuleList(GClist)
        for i in range(4):      # 0 1 2 3
            GCoutlist.append(nn.Sequential(nn.Conv2d(out_channels * 4, out_channel_list[i], 3, 1, 1),
                                           nn.ReLU(inplace=True),
                                           nn.Upsample(scale_factor=upsampe_scale[i], mode='bilinear')))
        self.GCoutmodel = nn.ModuleList(GCoutlist)

    def forward(self, x):
        xsize = x.size()[2:]
        global_context = []     # 0 1 2
        for i in range(len(self.GCmodule)):     # 0 1 2
            global_context.append(F.interpolate(self.GCmodule[i](x), xsize, mode='bilinear', align_corners=True))
        # global_context.append(self.GCmodule[-1](x))     # 0 1 2 3
        global_context = torch.cat(global_context, dim=1)

        output = []
        for i in range(len(self.GCoutmodel)):       # 0 1 2 3
            output.append(self.GCoutmodel[i](global_context))


        return output



""" Modify """
""" Global Context Module_RFB"""

class GCM_RFB_s(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCM_RFB_s, self).__init__()
        out_channel_list = [256, 128, 64, 64]
        upsampe_scale = [2, 4, 8, 16]
        GClist = []
        GCoutlist = []

        GClist.append(nn.Sequential(    # Conv1x1 -> Conv3x3, rate=1
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, dilation=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),

        ))
        GClist.append(nn.Sequential(    # Conv1x3 -> Conv3x3, rate=3
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3,1), padding=(1,0)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=3, dilation=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        ))
        GClist.append(nn.Sequential(    # Conv3x1 -> Conv3x3, rate=3
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=3, dilation=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        ))
        GClist.append(nn.Sequential(    # Conv5x5 -> Conv3x3, rate=5
            nn.Conv2d(in_channels, out_channels//2, 1),
            nn.BatchNorm2d(out_channels//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels//2, (out_channels//4)*3, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d((out_channels//4)*3),
            nn.ReLU(inplace=True),
            nn.Conv2d((out_channels // 4) * 3, out_channels, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=5, dilation=5),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        ))
        # GClist.append(nn.Sequential(            # Non-Local
        #     nn.Conv2d(in_channels, out_channels, 1, 1),
        #     nn.BatchNorm2d(out_channels),
        #     nn.ReLU(inplace=True),
        #     NonLocalBlock(out_channels)
        # ))
        self.GCmodule = nn.ModuleList(GClist)
        for i in range(4):      # 0 1 2 3
            GCoutlist.append(nn.Sequential(nn.Conv2d(out_channels * 4, out_channel_list[i], 3, 1, 1),
                                           nn.ReLU(inplace=True),
                                           nn.Upsample(scale_factor=upsampe_scale[i], mode='bilinear')))
        self.GCoutmodel = nn.ModuleList(GCoutlist)

    def forward(self, x):
        xsize = x.size()[2:]
        global_context = []     # 0 1 2
        for i in range(len(self.GCmodule)):     # 0 1 2
            global_context.append(F.interpolate(self.GCmodule[i](x), xsize, mode='bilinear', align_corners=True))
        # global_context.append(self.GCmodule[-1](x))     # 0 1 2 3
        global_context = torch.cat(global_context, dim=1)

        # output = []
        # for i in range(len(self.GCoutmodel)):       # 0 1 2 3
        #     output.append(self.GCoutmodel[i](global_context))

        return global_context




""" Global Context Module"""

class GCM(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=16, kernel_size=7):
        super(GCM, self).__init__()
        # pool_size = [1, 3, 5]
        pool_size = [1, 2, 4, 6]
        out_channel_list = [256, 128, 64, 64]
        upsampe_scale = [2, 4, 8, 16]
        GClist = []
        GCoutlist = []
        for ps in pool_size:
            GClist.append(nn.Sequential(        # Global average pool -> 3x3AdaptiveAvgPool -> 5x5AdaptiveAvgPool
                nn.AdaptiveAvgPool2d(ps),
                nn.Conv2d(in_channels, out_channels, 1, 1),
                nn.ReLU(inplace=True)))
        GClist.append(nn.Sequential(            # Non-Local
            nn.Conv2d(in_channels, out_channels, 1, 1),
            nn.ReLU(inplace=True),
            NonLocalBlock(out_channels)))
        self.GCmodule = nn.ModuleList(GClist)
        for i in range(4):      # 0 1 2 3
            GCoutlist.append(nn.Sequential(nn.Conv2d(out_channels * 4, out_channel_list[i], 3, 1, 1),
                                           nn.ReLU(inplace=True),
                                           nn.Upsample(scale_factor=upsampe_scale[i], mode='bilinear')))
        self.GCoutmodel = nn.ModuleList(GCoutlist)

        self.ca = ChannelAttention(channel=in_channels,reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        xsize = x.size()[2:]
        # residual = x
        # x = x * self.ca(x)
        global_context = []     # 0 1 2
        for i in range(len(self.GCmodule) - 1):     # 0 1 2
            global_context.append(F.interpolate(self.GCmodule[i](x), xsize, mode='bilinear', align_corners=True))
        global_context.append(self.GCmodule[-1](x))     # 0 1 2 3
        global_context = torch.cat(global_context, dim=1)
        # global_context = global_context * self.sa(global_context)

        # output = global_context + residual

        # output = []
        # for i in range(len(self.GCoutmodel)):       # 0 1 2 3
        #     output.append(self.GCoutmodel[i](global_context))

        return global_context



""" Adaptive Selection Module"""

class ASM(nn.Module):
    def __init__(self, in_channels, all_channels):
        super(ASM, self).__init__()
        self.non_local = NonLocalBlock(in_channels)
        self.selayer = SELayer(all_channels)

    def forward(self, lc, fuse, gc):
        fuse = self.non_local(fuse)
        fuse = torch.cat([lc, fuse, gc], dim=1)
        fuse = self.selayer(fuse)

        return fuse


""" Adaptive Selection Module"""


class ASM_nogcm(nn.Module):
    def __init__(self, in_channels, all_channels, lc_channel):
        super(ASM_nogcm, self).__init__()
        self.non_local = NonLocalBlock(in_channels)
        self.selayer = SELayer(all_channels)
        self.convl = nn.Conv2d(lc_channel, lc_channel * 2, kernel_size=1, stride=1)

    def forward(self, lc, fuse):
        fuse = self.non_local(fuse)
        lc = self.convl(lc)
        fuse = torch.cat([lc, fuse], dim=1)
        fuse = self.selayer(fuse)

        return fuse


"""
Squeeze and Excitation Layer

https://arxiv.org/abs/1709.01507

"""


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


"""
Non Local Block

https://arxiv.org/abs/1711.07971
"""

class NonLocalBlock(nn.Module):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NonLocalBlock, self).__init__()

        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                          kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                               kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, nn.MaxPool2d(kernel_size=(2, 2)))
            self.phi = nn.Sequential(self.phi, nn.MaxPool2d(kernel_size=(2, 2)))

    def forward(self, x):

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z


# # APF
# class up_(nn.Module, ABC):
#     def __init__(self, in_ch, out_ch, scale_factor=2):
#         super(up_, self).__init__()
#         self.up = nn.Sequential(
#             nn.Upsample(scale_factor=scale_factor),
#             nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True)
#         )
#
#     def forward(self, x):
#         return self.up(x)
#
# class ChannelAttention(nn.Module):
#     def __init__(self, channel, reduction=16):
#         super().__init__()
#         self.maxpool = nn.AdaptiveMaxPool2d(1)
#         self.avgpool = nn.AdaptiveAvgPool2d(1)
#         self.se = nn.Sequential(
#             nn.Conv2d(channel, channel // reduction, 1, bias=False),
#             nn.ReLU(),
#             nn.Conv2d(channel // reduction, channel, 1, bias=False)
#         )
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         max_result = self.maxpool(x)
#         avg_result = self.avgpool(x)
#         max_out = self.se(max_result)
#         avg_out = self.se(avg_result)
#         output = self.sigmoid(max_out + avg_out)
#         return output
#
# class APF(nn.Module, ABC):
#     def __init__(self, in_ch1, in_ch2, in_ch3, out_ch):
#         super(APF, self).__init__()
#         self.out_ch = out_ch
#         self.up2 = up_(in_ch2, out_ch, scale_factor=2)
#         self.up3 = up_(in_ch3, out_ch, scale_factor=4)
#         self.fusion = nn.Conv2d(out_ch * 4, out_ch, 1)
#         self.ca = ChannelAttention(out_ch, reduction=16)
#         self.gamma = nn.Parameter(torch.zeros(1))
#
#     def forward(self, x1, x2, x3):
#         x2 = self.up2(x2)
#         x3 = self.up3(x3)
#         x4 = torch.cat([x1, x2, x3], dim=1)
#         x4 = self.fusion(x4)
#         out = self.ca(x4)
#         # out = self.gamma * out + x4
#         # out = out * x4
#         return out


""" Global Context Module"""

class GCM_CBAM1(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=16, kernel_size=7):
        super(GCM_CBAM1, self).__init__()
        pool_size = [1, 2, 3, 5]
        out_channel_list = [256, 128, 64, 64]
        upsampe_scale = [2, 4, 8, 16]
        GClist = []
        GCoutlist = []
        for ps in pool_size:
            GClist.append(nn.Sequential(        # Global average pool -> 3x3AdaptiveAvgPool -> 5x5AdaptiveAvgPool
                nn.AdaptiveAvgPool2d(ps),
                nn.Conv2d(in_channels, out_channels, 1, 1),
                nn.ReLU(inplace=True))
            )
        # GClist.append(nn.Sequential(            # Non-Local
        #     nn.Conv2d(in_channels, out_channels, 1, 1),
        #     nn.ReLU(inplace=True),
        #     NonLocalBlock(out_channels)))
        self.GCmodule = nn.ModuleList(GClist)
        # for i in range(4):      # 0 1 2 3
        #     GCoutlist.append(nn.Sequential(nn.Conv2d(out_channels * 4, out_channel_list[i], 3, 1, 1),
        #                                    nn.ReLU(inplace=True),
        #                                    nn.Upsample(scale_factor=upsampe_scale[i], mode='bilinear')))
        # self.GCoutmodel = nn.ModuleList(GCoutlist)

        self.cbam = CSAMBlock(channel=out_channels)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x, lca):
        xsize = x.size()[2:]
        # residual = x
        # x = x * self.ca(x)
        global_context = []     # 0 1 2
        for i in range(len(self.GCmodule)):     # 0 1 2
            # global_context.append(F.interpolate(self.GCmodule[i](x), xsize, mode='bilinear', align_corners=True))
            x1 = self.GCmodule[i](x)
            x1 = F.interpolate(x1, xsize, mode='bilinear', align_corners=True)
            x1 = self.cbam(x1)
            global_context.append(x1)

        # global_context.append(self.GCmodule[-1](x))     # 0 1 2 3
        global_context = torch.cat(global_context, dim=1)
        lca = self.upsample(lca)
        global_context = global_context + lca
        # global_context = global_context * self.sa(global_context)

        # output = global_context + residual

        # output = []
        # for i in range(len(self.GCoutmodel)):       # 0 1 2 3
        #     output.append(self.GCoutmodel[i](global_context))

        return global_context


class GCM_CBAM23(nn.Module):
    def __init__(self, in_channels, out_channels, in_channel_lca, reduction=16, kernel_size=7):
        super(GCM_CBAM23, self).__init__()
        pool_size = [1, 2, 3, 5]
        out_channel_list = [256, 128, 64, 64]
        upsampe_scale = [2, 4, 8, 16]
        GClist = []
        GCoutlist = []
        for ps in pool_size:
            GClist.append(nn.Sequential(        # Global average pool -> 3x3AdaptiveAvgPool -> 5x5AdaptiveAvgPool
                nn.AdaptiveAvgPool2d(ps),
                nn.Conv2d(in_channels, out_channels, 1, 1),
                nn.ReLU(inplace=True))
            )
        # GClist.append(nn.Sequential(            # Non-Local
        #     nn.Conv2d(in_channels, out_channels, 1, 1),
        #     nn.ReLU(inplace=True),
        #     NonLocalBlock(out_channels)))
        self.GCmodule = nn.ModuleList(GClist)
        # for i in range(4):      # 0 1 2 3
        #     GCoutlist.append(nn.Sequential(nn.Conv2d(out_channels * 4, out_channel_list[i], 3, 1, 1),
        #                                    nn.ReLU(inplace=True),
        #                                    nn.Upsample(scale_factor=upsampe_scale[i], mode='bilinear')))
        # self.GCoutmodel = nn.ModuleList(GCoutlist)

        self.cbam = CSAMBlock(channel=out_channels)
        self.convl = nn.Conv2d(in_channel_lca, in_channel_lca // 2, kernel_size=1, stride=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x, lca):
        xsize = x.size()[2:]
        # residual = x
        # x = x * self.ca(x)
        global_context = []     # 0 1 2
        for i in range(len(self.GCmodule)):     # 0 1 2
            # global_context.append(F.interpolate(self.GCmodule[i](x), xsize, mode='bilinear', align_corners=True))
            x1 = self.GCmodule[i](x)
            x1 = F.interpolate(x1, xsize, mode='bilinear', align_corners=True)
            x1 = self.cbam(x1)
            global_context.append(x1)

        # global_context.append(self.GCmodule[-1](x))     # 0 1 2 3
        global_context = torch.cat(global_context, dim=1)
        lca = self.convl(lca)
        lca = self.upsample(lca)
        global_context = global_context + lca
        # global_context = global_context * self.sa(global_context)

        # output = global_context + residual

        # output = []
        # for i in range(len(self.GCoutmodel)):       # 0 1 2 3
        #     output.append(self.GCoutmodel[i](global_context))

        return global_context



class GCM_CBAM4(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=16, kernel_size=7):
        super(GCM_CBAM4, self).__init__()
        pool_size = [1, 2, 3, 5]
        out_channel_list = [256, 128, 64, 64]
        upsampe_scale = [2, 4, 8, 16]
        GClist = []
        GCoutlist = []
        for ps in pool_size:
            GClist.append(nn.Sequential(        # Global average pool -> 3x3AdaptiveAvgPool -> 5x5AdaptiveAvgPool
                nn.AdaptiveAvgPool2d(ps),
                nn.Conv2d(in_channels, out_channels, 1, 1),
                nn.ReLU(inplace=True))
            )
        # GClist.append(nn.Sequential(            # Non-Local
        #     nn.Conv2d(in_channels, out_channels, 1, 1),
        #     nn.ReLU(inplace=True),
        #     NonLocalBlock(out_channels)))
        self.GCmodule = nn.ModuleList(GClist)
        # for i in range(4):      # 0 1 2 3
        #     GCoutlist.append(nn.Sequential(nn.Conv2d(out_channels * 4, out_channel_list[i], 3, 1, 1),
        #                                    nn.ReLU(inplace=True),
        #                                    nn.Upsample(scale_factor=upsampe_scale[i], mode='bilinear')))
        # self.GCoutmodel = nn.ModuleList(GCoutlist)

        self.cbam = CSAMBlock(channel=out_channels)
        # self.convl = nn.Conv2d(in_channel_lca, in_channel_lca // 2, kernel_size=1, stride=1)
        # self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x):
        xsize = x.size()[2:]
        # residual = x
        # x = x * self.ca(x)
        global_context = []     # 0 1 2
        for i in range(len(self.GCmodule)):     # 0 1 2
            # global_context.append(F.interpolate(self.GCmodule[i](x), xsize, mode='bilinear', align_corners=True))
            x1 = self.GCmodule[i](x)
            x1 = F.interpolate(x1, xsize, mode='bilinear', align_corners=True)
            x1 = self.cbam(x1)
            global_context.append(x1)

        # global_context.append(self.GCmodule[-1](x))     # 0 1 2 3
        global_context = torch.cat(global_context, dim=1)
        # lca = self.upsample(lca)
        # global_context = global_context + lca
        # global_context = global_context * self.sa(global_context)

        # output = global_context + residual

        # output = []
        # for i in range(len(self.GCoutmodel)):       # 0 1 2 3
        #     output.append(self.GCoutmodel[i](global_context))

        return global_context



class GCM_CBAM(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=16, kernel_size=7):
        super(GCM_CBAM, self).__init__()
        pool_size = [1, 3, 5]
        out_channel_list = [256, 128, 64, 64]
        upsampe_scale = [2, 4, 8, 16]
        GClist = []
        GCoutlist = []
        for ps in pool_size:
            GClist.append(nn.Sequential(        # Global average pool -> 3x3AdaptiveAvgPool -> 5x5AdaptiveAvgPool
                nn.AdaptiveAvgPool2d(ps),
                nn.Conv2d(in_channels, out_channels, 1, 1),
                nn.ReLU(inplace=True))
            )
        GClist.append(nn.Sequential(            # Non-Local
            nn.Conv2d(in_channels, out_channels, 1, 1),
            nn.ReLU(inplace=True),
            NonLocalBlock(out_channels)))
        self.GCmodule = nn.ModuleList(GClist)
        for i in range(4):      # 0 1 2 3
            GCoutlist.append(nn.Sequential(nn.Conv2d(out_channels * 4, out_channel_list[i], 3, 1, 1),
                                           nn.ReLU(inplace=True),
                                           nn.Upsample(scale_factor=upsampe_scale[i], mode='bilinear')))
        self.GCoutmodel = nn.ModuleList(GCoutlist)

        self.cbam = CSAMBlock(channel=out_channels)


    def forward(self, x):
        xsize = x.size()[2:]

        global_context = []     # 0 1 2
        for i in range(len(self.GCmodule) - 1):     # 0 1 2
            # global_context.append(F.interpolate(self.GCmodule[i](x), xsize, mode='bilinear', align_corners=True))
            x1 = self.GCmodule[i](x)
            x1 = F.interpolate(x1, xsize, mode='bilinear', align_corners=True)
            x1 = self.cbam(x1)
            global_context.append(x1)

        x2 = self.GCmodule[-1](x)
        x2 = self.cbam(x2)
        global_context.append(x2)
        # global_context.append(self.GCmodule[-1](x))     # 0 1 2 3
        global_context = torch.cat(global_context, dim=1)
        # lca = self.upsample(lca)
        # global_context = global_context + lca
        # global_context = global_context * self.sa(global_context)

        # output = global_context + residual

        output = []
        for i in range(len(self.GCoutmodel)):       # 0 1 2 3
            output.append(self.GCoutmodel[i](global_context))

        return output


# 软池化 SoftPooling2D
class SoftPooling2D(torch.nn.Module):
    def __init__(self,kernel_size=3,strides=2,padding=1,ceil_mode = False,count_include_pad = True,divisor_override = None):
        super(SoftPooling2D, self).__init__()
        self.avgpool = torch.nn.AvgPool2d(kernel_size,strides,padding,ceil_mode,count_include_pad,divisor_override)
    def forward(self, x):
        x_exp = torch.exp(x)
        x_exp_pool = self.avgpool(x_exp)
        x = self.avgpool(x_exp*x)
        return x/x_exp_pool