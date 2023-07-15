import numpy as np
from skimage import io
import os
from abc import ABC

import torch
from torch import nn
from torch.nn import functional as F, init
from opt import opt
from models.ACSNet_caRABAsa_modDCR.modules import conv2d
from collections import OrderedDict


class SKAttention(nn.Module):

    def __init__(self, channel=512,kernels=[1,3,5,7],reduction=16,group=1,L=32):
        super().__init__()
        self.d=max(L,channel//reduction)
        self.convs=nn.ModuleList([])
        for k in kernels:
            self.convs.append(
                nn.Sequential(OrderedDict([
                    ('conv',nn.Conv2d(channel,channel,kernel_size=k,padding=k//2,groups=group)),
                    ('bn',nn.BatchNorm2d(channel)),
                    ('relu',nn.ReLU())
                ]))
            )
        self.fc=nn.Linear(channel,self.d)
        self.fcs=nn.ModuleList([])
        for i in range(len(kernels)):
            # self.fcs.append(nn.Linear(self.d,channel))
            self.fcs.append(nn.Conv2d(channel // reduction, channel, 1, bias=False))
        self.softmax=nn.Softmax(dim=0)

        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            # nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        # self.se1 = nn.Conv2d(channel // reduction, channel, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        bs, c, _, _ = x.size()
        conv_outs=[]
        ### split
        for conv in self.convs:
            conv_outs.append(conv(x))
        feats=torch.stack(conv_outs,0)#k,bs,channel,h,w

        ### fuse
        U=sum(conv_outs) #bs,c,h,w

        ### reduction channel
        # S=U.mean(-1).mean(-1) #bs,c
        # Z=self.fc(S) #bs,d
        max_result = self.maxpool(U)
        avg_result = self.avgpool(U)
        max_outZ = self.se(max_result)
        avg_outZ = self.se(avg_result)
        # Z = max_outZ + avg_outZ

        ### calculate attention weight
        weights=[]
        for fc in self.fcs:
            weight=fc(max_outZ) + fc(avg_outZ)
            weights.append(weight) #bs,channel
        attention_weughts=torch.stack(weights,0)#k,bs,channel,1,1
        attention_weughts=self.softmax(attention_weughts)#k,bs,channel,1,1

        ### fuse
        V=(attention_weughts*feats).sum(0)
        return V


class SoftPooling2D(torch.nn.Module):
    def __init__(self,kernel_size,strides=None,padding=0,ceil_mode = False,count_include_pad = True,divisor_override = None):
        super(SoftPooling2D, self).__init__()
        self.avgpool = torch.nn.AvgPool2d(kernel_size,strides,padding,ceil_mode,count_include_pad,divisor_override)
    def forward(self, x):
        x_exp = torch.exp(x)
        x_exp_pool = self.avgpool(x_exp)
        x = self.avgpool(x_exp*x)
        return x/x_exp_pool



class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):   # reduction8不如16
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


# 通道注意力 修改类似scSE
class ChannelAttention_mod(nn.Module):
    def __init__(self, channel, reduction=8):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.Conv_Squeeze = nn.Conv2d(channel, channel // reduction, kernel_size=1, bias=False)
        self.Conv_Excitation = nn.Conv2d(channel//reduction, channel, kernel_size=1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)

        max_out = self.Conv_Squeeze(max_result) # shape: [bs, c/2]   2,512,1,1
        max_out = self.Conv_Excitation(max_out) # shape: [bs, c]   2,1024,1,1

        avg_out = self.Conv_Squeeze(avg_result)  # shape: [bs, c/2]   2,512,1,1
        avg_out = self.Conv_Excitation(avg_out)  # shape: [bs, c]   2,1024,1,1

        output = self.sigmoid(max_out + avg_out)

        return x * output.expand_as(x)


# 通道注意力 修改类似ECA
class ChannelAttention_modeca(nn.Module):
    def __init__(self, channel, reduction=8, k_size = 3):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)

        max_out = self.conv(max_result.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        avg_out = self.conv(avg_result.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        output = self.sigmoid(max_out + avg_out)

        return x * output.expand_as(x)


# 通道注意力 修改GAM
class ChannelAttention_gam(nn.Module):
    def __init__(self, in_channels, rate = 16):
        super(ChannelAttention_gam, self).__init__()
        mid_channels = in_channels // rate
        self.channel_attention = nn.Sequential(
            nn.Linear(in_channels, mid_channels),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, in_channels)
        )

    def forward(self, x):
        # channel attention
        b, c, h, w = x.shape
        x_reshape = x.permute(0, 2, 3, 1).view(b, -1, c)
        x_att = self.channel_attention(x_reshape).view(b, h, w, c)
        x_channel_att = x_att.permute(0, 3, 1, 2)
        out = x * x_channel_att

        return out


# 通道注意力 修改GAM
class ChannelAttention_gam1(nn.Module):
    def __init__(self, in_channels, rate = 16):
        super(ChannelAttention_gam1, self).__init__()
        mid_channels = in_channels // rate
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.channel_attention = nn.Sequential(
            nn.Linear(in_channels, mid_channels),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, in_channels)
        )
        self.sigmoid = nn.Sigmoid()
        # self.l1 = nn.Linear(in_channels, mid_channels)
        # self.relu = nn.ReLU(inplace=True)
        # self.l2 = nn.Linear(mid_channels, in_channels)

    def forward(self, x):
        # channel attention
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        b, c, h, w = max_result.shape
        max_x_reshape = max_result.permute(0, 2, 3, 1).view(b, -1, c)
        max_x_att = self.channel_attention(max_x_reshape).view(b, h, w, c)
        x_max_att = max_x_att.permute(0, 3, 1, 2)

        avg_x_reshape = avg_result.permute(0, 2, 3, 1).view(b, -1, c)
        avg_x_att = self.channel_attention(avg_x_reshape).view(b, h, w, c)
        x_avg_att = avg_x_att.permute(0, 3, 1, 2)

        all_att = self.sigmoid(x_max_att + x_avg_att)

        out = x * all_att

        return out



# 通道注意力 修改GAM 通道+空间
class ChannelAttention_gam12(nn.Module):
    def __init__(self, in_channels, rate = 16):
        super(ChannelAttention_gam12, self).__init__()
        mid_channels = in_channels // rate
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.channel_attention = nn.Sequential(
            nn.Linear(in_channels, mid_channels),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, in_channels)
        )
        self.sigmoid = nn.Sigmoid()

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=7, padding=7 // 2),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, kernel_size=7, padding=7 // 2),
            nn.BatchNorm2d(in_channels),
        )

        # self.Conv1x1 = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
        # self.Conv1x3 = nn.Conv2d(in_channels, 1, kernel_size=3, bias=False)
        # self.Conv1x5 = nn.Conv2d(in_channels, 1, kernel_size=5, bias=False)
        # self.Conv1x7 = nn.Conv2d(in_channels, 1, kernel_size=7, bias=False)


    def forward(self, x):
        # channel attention
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        b, c, h, w = avg_result.shape
        max_x_reshape = max_result.permute(0, 2, 3, 1).view(b, -1, c)
        max_x_att = self.channel_attention(max_x_reshape).view(b, h, w, c)
        channel_x_max_att = max_x_att.permute(0, 3, 1, 2)

        avg_x_reshape = avg_result.permute(0, 2, 3, 1).view(b, -1, c)
        avg_x_att = self.channel_attention(avg_x_reshape).view(b, h, w, c)
        channel_x_avg_att = avg_x_att.permute(0, 3, 1, 2)

        channel_all_att = self.sigmoid(channel_x_max_att + channel_x_avg_att)
        channel_out = x * channel_all_att

        x_spatial_att = self.spatial_attention(channel_out)
        x_spatial_att = self.sigmoid(x_spatial_att)

        out = channel_out * x_spatial_att
        out = out + x
        return out



# 通道注意力 修改GAM 通道+空间
class ChannelAttention_gam123(nn.Module):
    def __init__(self, in_channels, rate = 16):
        super(ChannelAttention_gam123, self).__init__()
        mid_channels = in_channels // rate
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.channel_attention = nn.Sequential(
            nn.Linear(in_channels, mid_channels),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, in_channels)
        )
        self.sigmoid = nn.Sigmoid()

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=7, padding=7 // 2),
        )

    def forward(self, x):
        # channel attention
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        b, c, h, w = avg_result.shape
        max_x_reshape = max_result.permute(0, 2, 3, 1).view(b, -1, c)
        max_x_att = self.channel_attention(max_x_reshape).view(b, h, w, c)
        channel_x_max_att = max_x_att.permute(0, 3, 1, 2)

        avg_x_reshape = avg_result.permute(0, 2, 3, 1).view(b, -1, c)
        avg_x_att = self.channel_attention(avg_x_reshape).view(b, h, w, c)
        channel_x_avg_att = avg_x_att.permute(0, 3, 1, 2)

        channel_all_att = self.sigmoid(channel_x_max_att + channel_x_avg_att)
        channel_out = x * channel_all_att

        x_spatial_att = self.spatial_attention(channel_out)
        x_spatial_att = self.sigmoid(x_spatial_att)

        out = channel_out * x_spatial_att
        out = out + x

        return out






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

    def __init__(self, channel, reduction=16, kernel_size=49):
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


#
# """ Reverse Attention Module"""
#
# class RA1(nn.Module):
#     def __init__(self, in_channel, reduction=16, kernel_size=7):
#         super(RA1, self).__init__()
#         self.channel = in_channel
#         # self.convl = nn.Conv2d(in_channel_lca, in_channel_lca // 2, kernel_size=1, stride=1)
#         self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
#
#         self.ca = ChannelAttention(channel=in_channel,reduction=reduction)
#         self.sa = SpatialAttention(kernel_size=kernel_size)
#         self.cbam = CSAMBlock(channel= 64)
#         self.g = conv2d(64 + 64, 64, 1)
#
#         self.selayer = SELayer(in_channel)
#
#     def forward(self, x, pred, lca):
#         residual = x
#         # x = self.cbam(x)
#         x = x * self.ca(x)
#         # RA
#         a = torch.sigmoid(-pred)
#         # x = self.relu(self.bn(self.convert(x)))
#         att_ra = a.expand(-1, self.channel, -1, -1).mul(x)  # a与x逐元素乘法
#         # LCA
#         score = torch.sigmoid(pred)
#         dist = torch.abs(score - 0.5)
#         att1 = 1 - (dist / 0.5)
#         att_lca = x * att1
#
#         attall = att_ra + att_lca
#         # attall = self.cbam(attall)
#
#         lca = self.upsample(lca)
#         attall = attall + lca
#         # att1 = att * self.ca(att)
#         attall = attall * self.sa(attall)
#         # att = torch.cat([att1, att2], dim=1)
#         # att = att1 + att2
#
#         # y = attall + residual + lca
#         y = attall + residual
#         return y
#
#
#
# """ Local Context Attention Module"""
#
# class RA2(nn.Module):
#     def __init__(self, in_channel, in_channel_lca, reduction=16, kernel_size=7):
#         super(RA2, self).__init__()
#         self.channel = in_channel
#         self.convl = nn.Conv2d(in_channel_lca, in_channel_lca // 2, kernel_size=1, stride=1)
#         self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
#
#         self.ca = ChannelAttention(channel=in_channel,reduction=reduction)
#         self.sa = SpatialAttention(kernel_size=kernel_size)
#         self.selayer = SELayer(in_channel)
#
#         self.cbam = CSAMBlock(channel=64)
#         self.g = conv2d(64 + 64, 64, 1)
#
#     def forward(self, x, pred, lca):
#         residual = x
#         # x = self.cbam(x)
#         x = x * self.ca(x)
#         # RA
#         a = torch.sigmoid(-pred)
#         # x = self.relu(self.bn(self.convert(x)))
#         att_ra = a.expand(-1, self.channel, -1, -1).mul(x)  # a与x逐元素乘法
#         # LCA
#         score = torch.sigmoid(pred)
#         dist = torch.abs(score - 0.5)
#         att1 = 1 - (dist / 0.5)
#         att_lca = x * att1
#
#         lca = self.convl(lca)
#         lca = self.upsample(lca)
#
#         attall = att_ra + att_lca
#         attall = attall + lca
#         attall = attall * self.sa(attall)
#         # attall = self.cbam(attall)
#
#
#
#         # y = attall + residual + lca
#         y = attall + residual
#         return y
#
#
#
# """ Local Context Attention Module"""
#
# class RA3(nn.Module):
#     def __init__(self, in_channel, in_channel_lca, reduction=16, kernel_size=7):
#         super(RA3, self).__init__()
#         self.channel = in_channel
#         self.convl = nn.Conv2d(in_channel_lca, in_channel_lca // 2, kernel_size=1, stride=1)
#         self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
#
#         self.ca = ChannelAttention(channel=in_channel,reduction=reduction)
#         self.sa = SpatialAttention(kernel_size=kernel_size)
#         self.selayer = SELayer(in_channel)
#
#         self.cbam = CSAMBlock(channel=128)
#         self.g = conv2d(128 + 128, 128, 1)
#
#     def forward(self, x, pred, lca):
#         residual = x
#         # x = self.cbam(x)
#         x = x * self.ca(x)
#         # RA
#         a = torch.sigmoid(-pred)
#         # x = self.relu(self.bn(self.convert(x)))
#         att_ra = a.expand(-1, self.channel, -1, -1).mul(x)  # a与x逐元素乘法
#         # LCA
#         score = torch.sigmoid(pred)
#         dist = torch.abs(score - 0.5)
#         att1 = 1 - (dist / 0.5)
#         att_lca = x * att1
#
#         lca = self.convl(lca)
#         lca = self.upsample(lca)
#
#         attall = att_ra + att_lca
#         attall = attall + lca
#         attall = attall * self.sa(attall)
#         # attall = self.cbam(attall)
#
#
#         y = attall + residual + lca
#         return y
#
#
# """ Local Context Attention Module"""
#
# class RA4(nn.Module):
#     def __init__(self, in_channel, reduction=16, kernel_size=7):
#         super(RA4, self).__init__()
#         self.channel = in_channel
#
#         self.ca = ChannelAttention(channel=in_channel,reduction=reduction)
#         self.sa = SpatialAttention(kernel_size=kernel_size)
#
#         self.cbam = CSAMBlock(channel = 256)
#         self.g = conv2d(256 + 256, 256, 1)
#
#     def forward(self, x, pred):
#         residual = x
#         # x = self.cbam(x)
#         x = x * self.ca(x)
#         # x = x * self.sa(x)
#         # RA
#         a = torch.sigmoid(-pred)
#         # x = self.relu(self.bn(self.convert(x)))
#         att_ra = a.expand(-1, self.channel, -1, -1).mul(x)       # a与x逐元素乘法
#
#         # att_ra = self.sideout(att_ra)
#         # att_ra = torch.sigmoid(att_ra)
#         # att_ra = att_ra.squeeze(att_ra[0]).detach().cpu().numpy()
#         # att_ra[att_ra > 0.5] = 255
#         # att_ra[att_ra <= 0.5] = 0
#         # io.imsave(os.path.join("./result11/" + "att_ra" + str(i) + ".jpg"), att_ra)
#
#         #LCA
#         score = torch.sigmoid(pred)
#         dist = torch.abs(score - 0.5)
#         att1 = 1 - (dist / 0.5)
#         att_lca = x * att1
#
#         attall = att_ra + att_lca
#         attall = attall * self.sa(attall)
#         # attall = self.cbam(attall)
#
#
#         # att_x_cbam = torch.cat([x_cbam, attall], 1)
#         # att_x_cbam = self.g(att_x_cbam)
#         # att = att * self.sa(att)
#         y = attall + residual
#
#         return y ca-RA+LCA + 上一层lca - sa
# ca-RA+LCA + 上一层lca - sa

""" Reverse Attention Module"""

class RA1(nn.Module):
    def __init__(self, in_channel, reduction=16, kernel_size=7):
        super(RA1, self).__init__()
        self.channel = in_channel
        # self.convl = nn.Conv2d(in_channel_lca, in_channel_lca // 2, kernel_size=1, stride=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

        self.ca = ChannelAttention(channel=in_channel,reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)
        self.cbam = CSAMBlock(channel= 64)
        self.g = conv2d(64 + 64, 64, 1)

        self.selayer = SELayer(in_channel)

        self.out = nn.Sequential(
            conv2d(64, 32, 1),
            nn.Conv2d(32, 1, 1)
        )

    def forward(self, x, pred, lca):
        residual = x
        # residual1 = self.out(residual)
        # residual_out = torch.squeeze(residual1).cpu().numpy()  # 320,320
        # print(residual_out)
        # io.imsave(f"./aaa/{opt.model}/old1.jpg", residual_out)

        # pred1 = torch.sigmoid(pred)
        # pred1 = torch.squeeze(pred1).cpu().numpy()  # 320,320
        # print(pred1)
        # io.imsave(f"./aaa/{opt.model}/pred1.jpg", pred1)

        # x = x * self.ca(x)
        # x1 = self.out(x)
        # x1_out = torch.squeeze(x1).cpu().numpy()  # 320,320
        # print(x1_out)
        # io.imsave(f"./aaa/{opt.model}/ca1.jpg", x1_out)

        # RA
        a = torch.sigmoid(-pred) # 0.9
        # a1_out = torch.squeeze(a).cpu().numpy()  # 320,320
        # print(a1_out)
        # io.imsave(f"./aaa/{opt.model}/reverse1.jpg", a1_out)


        att_ra = a.expand(-1, self.channel, -1, -1).mul(x)  # a与x逐元素乘法
        # # att_ra = a * x
        # att_ra1 = self.out(att_ra)
        # # att_lca1 = torch.sigmoid(att_lca1)
        # att_ra1_out = torch.squeeze(att_ra1).cpu().numpy()  # 320,320
        # print(att_ra1_out)
        # io.imsave(f"./aaa/{opt.model}/ra_out1.jpg", att_ra1_out)


        # LCA
        score = torch.sigmoid(pred) #0.1
        dist = torch.abs(score - 0.5)
        att1 = 1 - (dist / 0.5) #0.2    0.3
        # att1_out = torch.squeeze(att1).cpu().numpy()  # 320,320
        # print(att1_out)
        # io.imsave(f"./aaa/{opt.model}/lca1.jpg", att1_out)

        att_lca = x * att1
        # att_lca1 = self.out(att_lca)
        # # att_lca1 = torch.sigmoid(att_lca1)
        # att_lca1_out = torch.squeeze(att_lca1).cpu().numpy()  # 320,320
        # print(att_lca1_out)
        # io.imsave(f"./aaa/{opt.model}/lca_out1.jpg", att_lca1_out)


        # attall = att_lca
        attall = torch.cat([att_ra,att_lca], dim=1)
        attall = self.g(attall)
        # attall1 = self.out(attall)
        # attall1_out = torch.squeeze(attall1).cpu().numpy()  # 320,320
        # print(attall1_out)
        # io.imsave(f"./aaa/{opt.model}/ra_lca1.jpg", attall1_out)


        lca = self.upsample(lca)
        # lca1 = self.out(lca)
        # lca1_out = torch.squeeze(lca1).cpu().numpy()  # 320,320
        # print(lca1_out)
        # io.imsave(f"./aaa/{opt.model}/lca_last1.jpg", lca1_out)
        #

        attall = attall + lca
        # attall_lca = self.out(attall)
        # attall_lca_out = torch.squeeze(attall_lca).cpu().numpy()  # 320,320
        # print(attall_lca_out)
        # io.imsave(f"./aaa/{opt.model}/ra_lca_lcalast1.jpg", attall_lca_out)


        # attall = attall * self.sa(attall)
        attall = attall * self.ca(attall)
        # attallall = self.out(attall)
        # attallall_out = torch.squeeze(attallall).cpu().numpy()  # 320,320
        # print(attallall_out)
        # io.imsave(f"./aaa/{opt.model}/ra_lca_lcalast_sa1.jpg", attallall_out)


        y = attall + residual
        # y1 = self.out(y)
        # # y1 = torch.sigmoid(y1)
        # y1_out = torch.squeeze(y1).cpu().numpy()  # 320,320
        # print(y1_out)
        # io.imsave(f"./aaa/{opt.model}/outall1.jpg", y1_out)



        return y



""" Local Context Attention Module"""

class RA2(nn.Module):
    def __init__(self, in_channel, in_channel_lca, reduction=16, kernel_size=7):
        super(RA2, self).__init__()
        self.channel = in_channel
        self.convl = nn.Conv2d(in_channel_lca, in_channel_lca // 2, kernel_size=1, stride=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

        self.ca = ChannelAttention(channel=in_channel,reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)
        self.selayer = SELayer(in_channel)

        self.cbam = CSAMBlock(channel=64)
        self.g = conv2d(64 + 64, 64, 1)

        self.out = nn.Sequential(
            conv2d(64, 32, 1),
            nn.Conv2d(32, 1, 1)
        )

    def forward(self, x, pred, lca):
        residual = x
        # residual1 = self.out(residual)
        # residual1 = torch.sigmoid(residual1)
        # residual_out = torch.squeeze(residual1).cpu().numpy()  # 320,320
        # print(residual_out)
        # io.imsave(f"./aaa/{opt.model}/old2.jpg", residual_out)

        # pred1 = torch.sigmoid(pred)
        # pred1 = torch.squeeze(pred1).cpu().numpy()  # 320,320
        # print(pred1)
        # io.imsave(f"./aaa/{opt.model}/pred2.jpg", pred1)


        # x = x * self.ca(x)
        # x1 = self.out(x)
        # x1_out = torch.squeeze(x1).cpu().numpy()  # 320,320
        # print(x1_out)
        # io.imsave(f"./aaa/{opt.model}/ca2.jpg", x1_out)

        # RA
        a = torch.sigmoid(-pred)
        # a1_out = torch.squeeze(a).cpu().numpy()  # 320,320
        # print(a1_out)
        # io.imsave(f"./aaa/{opt.model}/reverse2.jpg", a1_out)


        att_ra = a.expand(-1, self.channel, -1, -1).mul(x)  # a与x逐元素乘法
        # att_ra1 = self.out(att_ra)
        # att_ra1_out = torch.squeeze(att_ra1).cpu().numpy()  # 320,320
        # print(att_ra1_out)
        # io.imsave(f"./aaa/{opt.model}/ra_out2.jpg", att_ra1_out)


        # LCA
        score = torch.sigmoid(pred)
        dist = torch.abs(score - 0.5)
        att1 = 1 - (dist / 0.5)
        # att1_out = torch.squeeze(att1).cpu().numpy()  # 320,320
        # print(att1_out)
        # io.imsave(f"./aaa/{opt.model}/lca2.jpg", att1_out)



        att_lca = x * att1
        # att_lca1 = self.out(att_lca)
        # att_lca1_out = torch.squeeze(att_lca1).cpu().numpy()  # 320,320
        # print(att_lca1_out)
        # io.imsave(f"./aaa/{opt.model}/lca_out2.jpg", att_lca1_out)



        lca = self.convl(lca)
        lca = self.upsample(lca)
        # lca1 = self.out(lca)
        # lca1_out = torch.squeeze(lca1).cpu().numpy()  # 320,320
        # print(lca1_out)
        # io.imsave(f"./aaa/{opt.model}/lca_last2.jpg", lca1_out)


        # attall = att_lca
        attall = torch.cat([att_ra,att_lca], dim=1)
        attall = self.g(attall)
        # attall1 = self.out(attall)
        # attall1_out = torch.squeeze(attall1).cpu().numpy()  # 320,320
        # print(attall1_out)
        # io.imsave(f"./aaa/{opt.model}/ra_lca2.jpg", attall1_out)


        attall = attall + lca
        # attall_lca = self.out(attall)
        # attall_lca_out = torch.squeeze(attall_lca).cpu().numpy()  # 320,320
        # print(attall_lca_out)
        # io.imsave(f"./aaa/{opt.model}/ra_lca_lcalast2.jpg", attall_lca_out)



        # attall = attall * self.sa(attall)
        attall = attall * self.ca(attall)
        # attallall = self.out(attall)
        # attallall_out = torch.squeeze(attallall).cpu().numpy()  # 320,320
        # print(attallall_out)
        # io.imsave(f"./aaa/{opt.model}/ra_lca_lcalast_sa2.jpg", attallall_out)



        y = attall + residual
        # y1 = self.out(y)
        # y1_out = torch.squeeze(y1).cpu().numpy()  # 320,320
        # print(y1_out)
        # io.imsave(f"./aaa/{opt.model}/outall22.jpg", y1_out)



        return y



""" Local Context Attention Module"""

class RA3(nn.Module):
    def __init__(self, in_channel, in_channel_lca, reduction=16, kernel_size=7):
        super(RA3, self).__init__()
        self.channel = in_channel
        self.convl = nn.Conv2d(in_channel_lca, in_channel_lca // 2, kernel_size=1, stride=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

        self.ca = ChannelAttention(channel=in_channel,reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)
        self.selayer = SELayer(in_channel)

        self.cbam = CSAMBlock(channel=128)
        self.g = conv2d(128 + 128, 128, 1)

    def forward(self, x, pred, lca):
        residual = x

        # pred1 = torch.sigmoid(pred)
        # pred1 = torch.squeeze(pred1).cpu().numpy()  # 320,320
        # print(pred1)
        # io.imsave(f"./aaa/{opt.model}/pred3.jpg", pred1)


        # x = self.cbam (x)
        # x = x * self.ca(x)
        # RA
        a = torch.sigmoid(-pred)
        att_ra = a.expand(-1, self.channel, -1, -1).mul(x)  # a与x逐元素乘法
        # LCA
        score = torch.sigmoid(pred)
        dist = torch.abs(score - 0.5)
        att1 = 1 - (dist / 0.5)
        att_lca = x * att1

        lca = self.convl(lca)
        lca = self.upsample(lca)

        # attall = att_lca
        attall = torch.cat([att_ra,att_lca], dim=1)
        attall = self.g(attall)
        attall = attall + lca
        # attall = attall * self.sa(attall)
        attall = attall * self.ca(attall)

        y = attall + residual

        return y


""" Local Context Attention Module"""

class RA4(nn.Module):
    def __init__(self, in_channel, in_channel_lca, reduction=16, kernel_size=7):
        super(RA4, self).__init__()
        self.channel = in_channel

        self.convl = nn.Conv2d(in_channel_lca, in_channel_lca // 2, kernel_size=1, stride=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

        self.ca = ChannelAttention(channel=in_channel,reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)
        self.selayer = SELayer(in_channel)

        self.cbam = CSAMBlock(channel = 256)
        self.g = conv2d(256 + 256, 256, 1)

        self.out = nn.Sequential(
            conv2d(256, 64, 1),
            nn.Conv2d(64, 1, 1)
        )

    def forward(self, x, pred, lca):
        residual = x

        # pred1 = torch.sigmoid(pred)
        # pred1 = torch.squeeze(pred1).cpu().numpy()  # 320,320
        # print(pred1)
        # io.imsave(f"./aaa/{opt.model}/pred4.jpg", pred1)


        # x = x * self.ca(x)
        #RA
        a = torch.sigmoid(-pred)
        att_ra = a.expand(-1, self.channel, -1, -1).mul(x)       # a与x逐元素乘法


        #LCA
        score = torch.sigmoid(pred)
        dist = torch.abs(score - 0.5)
        att1 = 1 - (dist / 0.5)


        att_lca = x * att1

        # attall = att_lca
        attall = torch.cat([att_ra, att_lca], dim=1)
        attall = self.g(attall)

        lca = self.convl(lca)
        lca = self.upsample(lca)

        attall = attall + lca

        # attall = attall * self.sa(attall)
        attall = attall * self.ca(attall)

        y = attall + residual

        return y




def spatial_kernel_size(input, beta=1, gamma=2):
    """
    The original implementation based the kernel size on the channel, but with
    square images, it can be simplified to:
    k = | log2(R)/gamma + b/gamma|odd
    """

    k = int(abs(np.log2((input.shape[1])/gamma) + beta/gamma))
    out = k if k % 2 else k + 1

    return out


class GCM_RFB_non(nn.Module):
    def __init__(self):
        super(GCM_RFB_non, self).__init__()
        self.pool1 = nn.Sequential(  # Global average pool -> 3x3AdaptiveAvgPool -> 5x5AdaptiveAvgPool
            nn.Conv2d(512, 128, 3, padding=1, dilation=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True))
        self.pool3 = nn.Sequential(  # Global average pool -> 3x3AdaptiveAvgPool -> 5x5AdaptiveAvgPool
            nn.Conv2d(512, 128, 3, padding=3, dilation=3),
            nn.BatchNorm2d(128),
            nn.ReLU(True))
        self.pool5 = nn.Sequential(  # Global average pool -> 3x3AdaptiveAvgPool -> 5x5AdaptiveAvgPool
            nn.Conv2d(512, 128, 3, padding=5, dilation=5),
            nn.BatchNorm2d(128),
            nn.ReLU(True))
        # self.pool7 = nn.Sequential(
        #     nn.Conv2d(512, 128, 3, padding=7, dilation=7),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(True))
        self.pool7 = nn.Sequential(            # Non-Local
            nn.Conv2d(512, 128, 1, 1),
            nn.ReLU(inplace=True),
            NonLocalBlock(128))


        self.L = nn.Conv2d(512, 1, 1)
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=7 // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        X1 = self.pool1(x)
        # max_result1, _ = torch.max(X1, dim=1, keepdim=True)
        # avg_result1 = torch.mean(X1, dim=1, keepdim=True)
        # result1 = torch.cat([max_result1, avg_result1], 1)
        # output1 = self.conv(result1)
        # output1 = self.sigmoid(output1)
        # output1 = X1 * output1

        X2 = self.pool3(x)
        # max_result2, _ = torch.max(X2, dim=1, keepdim=True)
        # avg_result2 = torch.mean(X2, dim=1, keepdim=True)
        # result2 = torch.cat([max_result2, avg_result2], 1)
        # output2 = self.conv(result2)
        # output2 = self.sigmoid(output2)
        # output2 = X2 * output2

        X3 = self.pool5(x)
        # max_result3, _ = torch.max(X3, dim=1, keepdim=True)
        # avg_result3 = torch.mean(X3, dim=1, keepdim=True)
        # result3 = torch.cat([max_result3, avg_result3], 1)
        # output3 = self.conv(result3)
        # output3 = self.sigmoid(output3)
        # output3 = X3 * output3

        X4 = self.pool7(x)
        # max_result4, _ = torch.max(X4, dim=1, keepdim=True)
        # avg_result4 = torch.mean(X4, dim=1, keepdim=True)
        # result4 = torch.cat([max_result4, avg_result4], 1)
        # output4 = self.conv(result4)
        # output4 = self.sigmoid(output4)
        # output4 = X4 * output4

        # output = torch.cat([output1,output2,output3,output4], dim=1)

        gcm = torch.cat([X1,X2,X3,X4], dim=1)
        max_result, _ = torch.max(gcm, dim=1, keepdim=True)
        avg_result = torch.mean(gcm, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output)

        output = output * x
        out = output + x

        return out





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

        # self.ca = ChannelAttention(channel=in_channels,reduction=reduction)
        # self.sa = SpatialAttention(kernel_size=kernel_size)

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
        self.g = conv2d(in_channels//2 , in_channels, 1)

    def forward(self, lc, fuse, gc):
        fuse = self.non_local(fuse)
        fuse = torch.cat([lc, fuse, gc], dim=1)
        # gc = self.g(gc)
        # fuse = torch.cat([fuse, gc], dim=1)
        fuse = self.selayer(fuse)

        return fuse

# class ASMend(nn.Module):
#     def __init__(self, in_channels, all_channels):
#         super(ASMend, self).__init__()
#         self.non_local = NonLocalBlock(in_channels)
#         self.selayer = SELayer(all_channels)
#
#     def forward(self, lc, fuse, gc):
#         fuse = self.non_local(fuse)
#         # fuse = torch.cat([lc, fuse, gc], dim=1)
#         fuse = torch.cat([fuse, gc], dim=1)
#         fuse = self.selayer(fuse)
#
#         return fuse


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







class SpatialAttention1(nn.Module):
    def __init__(self, kernel_size=4):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 1)
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


class ASMcasa(nn.Module):
    def __init__(self, in_channels, rate = 4):
        super(ASMcasa, self).__init__()
        self.non_local = NonLocalBlock(in_channels)
        self.convl = nn.Conv2d(in_channels // 2,in_channels, kernel_size=1, stride=1)
        self.ca = ChannelAttention(in_channels)
        # self.sa = SpatialAttention(kernel_size=7)
        self.convout = nn.Conv2d(in_channels, in_channels*2 , kernel_size=3, stride=1, padding=1)


    def forward(self, lc, fuse, gc):
        fuse = self.non_local(fuse)

        lc = self.convl(lc)
        gc = self.convl(gc)

        fuse1 = fuse + lc
        fuse1 = self.ca(fuse1) * fuse1
        fuse2 = fuse + gc
        fuse2 = self.ca(fuse2) * fuse2
        fuse_all = torch.cat([fuse1, fuse2], dim= 1)
        # fuse = self.convout(fuse)
        # fuse_all = fuse_all + fuse

        # fuse_all = fuse+lc+gc
        # fuse_all = self.convout(fuse_all)


        return fuse_all


class ASMcasa1(nn.Module):
    def __init__(self, in_channels, rate = 4):
        super(ASMcasa1, self).__init__()
        self.non_local = NonLocalBlock(in_channels)
        self.convl = nn.Conv2d(in_channels // 2,in_channels, kernel_size=1, stride=1)
        self.convout = nn.Conv2d(128,192, kernel_size=1, stride=1)
        self.ca = ChannelAttention(in_channels)
        # self.sa = SpatialAttention(kernel_size=7)
        self.convout1 = nn.Conv2d(in_channels, in_channels*2 , kernel_size=3, stride=1, padding=1)


    def forward(self, lc, fuse, gc):
        fuse = self.non_local(fuse)

        fuse1 = fuse + lc
        fuse1 = self.ca(fuse1) * fuse1
        fuse2 = fuse + gc
        fuse2 = self.ca(fuse2) * fuse2
        fuse_all = torch.cat([fuse1, fuse2], dim= 1)
        # fuse = self.convout1(fuse)
        # fuse_all = fuse_all + fuse
        fuse_all = self.convout(fuse_all)

        # fuse_all = fuse+lc+gc
        # fuse_all = self.convout1(fuse_all)

        return fuse_all