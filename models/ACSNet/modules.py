import os

import skimage.io as io

import torch
from torch import nn
from torch.nn import functional as F

from models.ACSNet_DCR.modules import conv2d
from opt import opt

""" Local Context Attention Module"""

class LCA(nn.Module):
    def __init__(self):
        super(LCA, self).__init__()

        self.out = nn.Sequential(
            conv2d(64, 32, 1),
            nn.Conv2d(32, 1, 1)
        )

    def forward(self, x, pred):
        residual = x
        score = torch.sigmoid(pred)
        dist = torch.abs(score - 0.5)
        att = 1 - (dist / 0.5)

        att1_out = torch.squeeze(att).cpu().numpy()  # 320,320
        print(att1_out)
        io.imsave(f"./aaa/{opt.model}/lca.jpg", att1_out)

        att_x = x * att

        att_lca1 = self.out(att_x)
        att_lca1 = torch.sigmoid(att_lca1)
        att_lca1_out = torch.squeeze(att_lca1).cpu().numpy()  # 320,320
        print(att_lca1_out)
        io.imsave(f"./aaa/{opt.model}/lca_out.jpg", att_lca1_out)



        out = att_x + residual

        return out

class LCA1(nn.Module):
    def __init__(self):
        super(LCA1, self).__init__()

        self.out = nn.Sequential(
            conv2d(64, 32, 1),
            nn.Conv2d(32, 1, 1)
        )

    def forward(self, x, pred):
        residual = x
        # residual1 = self.out(residual)
        # residual_out = torch.squeeze(residual1).cpu().numpy()  # 320,320
        # print(residual_out)
        # io.imsave(f"./aaa/{opt.model}/old1.jpg", residual_out)


        score = torch.sigmoid(pred)
        # score_out = torch.squeeze(score).cpu().numpy()  # 320,320
        # print(score_out)
        # io.imsave(f"./aaa/{opt.model}/predict1.jpg", score_out)

        dist = torch.abs(score - 0.5)
        att = 1 - (dist / 0.5)
        # att1_out = torch.squeeze(att).cpu().numpy()  # 320,320
        # print(att1_out)
        # io.imsave(f"./aaa/{opt.model}/lca1.jpg", att1_out)



        att_x = x * att
        #
        # att_lca1 = self.out(att_x)
        # # att_lca1 = torch.sigmoid(att_lca1)
        # att_lca1_out = torch.squeeze(att_lca1).cpu().numpy()  # 320,320
        # print(att_lca1_out)
        # io.imsave(f"./aaa/{opt.model}/lca_out1.jpg", att_lca1_out)



        out = att_x + residual
        #
        # out1 = self.out(out)
        # # out1 = torch.sigmoid(out1)
        # out1 = torch.squeeze(out1).cpu().numpy()  # 320,320
        # print(out1)
        # io.imsave(f"./aaa/{opt.model}/lca_out_residual1.jpg", out1)


        return out

class LCA2(nn.Module):
    def __init__(self):
        super(LCA2, self).__init__()

        self.out = nn.Sequential(
            conv2d(64, 32, 1),
            nn.Conv2d(32, 1, 1)
        )

    def forward(self, x, pred):
        residual = x
        # residual1 = self.out(residual)
        # residual_out = torch.squeeze(residual1).cpu().numpy()  # 320,320
        # print(residual_out)
        # io.imsave(f"./aaa/{opt.model}/old2.jpg", residual_out)

        score = torch.sigmoid(pred)
        # score_out = torch.squeeze(score).cpu().numpy()  # 320,320
        # print(score_out)
        # io.imsave(f"./aaa/{opt.model}/predict2.jpg", score_out)

        dist = torch.abs(score - 0.5)
        att = 1 - (dist / 0.5)
        # att1_out = torch.squeeze(att).cpu().numpy()  # 320,320
        # print(att1_out)
        # io.imsave(f"./aaa/{opt.model}/lca2.jpg", att1_out)

        att_x = x * att

        # att_lca1 = self.out(att_x)
        # # att_lca1 = torch.sigmoid(att_lca1)
        # att_lca1_out = torch.squeeze(att_lca1).cpu().numpy()  # 320,320
        # print(att_lca1_out)
        # io.imsave(f"./aaa/{opt.model}/lca_out2.jpg", att_lca1_out)

        out = att_x + residual

        # out1 = self.out(out)
        # # out1 = torch.sigmoid(out1)
        # out1 = torch.squeeze(out1).cpu().numpy()  # 320,320
        # print(out1)
        # io.imsave(f"./aaa/{opt.model}/lca_out_residual2.jpg", out1)

        return out

class LCA3(nn.Module):
    def __init__(self):
        super(LCA3, self).__init__()

        self.out = nn.Sequential(
            conv2d(128, 32, 1),
            nn.Conv2d(32, 1, 1)
        )

    def forward(self, x, pred):
        residual = x
        # residual1 = self.out(residual)
        # residual_out = torch.squeeze(residual1).cpu().numpy()  # 320,320
        # print(residual_out)
        # io.imsave(f"./aaa/{opt.model}/old3.jpg", residual_out)

        score = torch.sigmoid(pred)
        # score_out = torch.squeeze(score).cpu().numpy()  # 320,320
        # print(score_out)
        # io.imsave(f"./aaa/{opt.model}/predict3.jpg", score_out)

        dist = torch.abs(score - 0.5)
        att = 1 - (dist / 0.5)
        # att1_out = torch.squeeze(att).cpu().numpy()  # 320,320
        # print(att1_out)
        # io.imsave(f"./aaa/{opt.model}/lca3.jpg", att1_out)



        att_x = x * att
        # att_lca1 = self.out(att_x)
        # # att_lca1 = torch.sigmoid(att_lca1)
        # att_lca1_out = torch.squeeze(att_lca1).cpu().numpy()  # 320,320
        # print(att_lca1_out)
        # io.imsave(f"./aaa/{opt.model}/lca_out3.jpg", att_lca1_out)



        out = att_x + residual
        #
        # out1 = self.out(out)
        # # att_lca1 = torch.sigmoid(att_lca1)
        # out1 = torch.squeeze(out1).cpu().numpy()  # 320,320
        # print(out1)
        # io.imsave(f"./aaa/{opt.model}/lca_out_residual3.jpg", out1)


        return out

class LCA4(nn.Module):
    def __init__(self):
        super(LCA4, self).__init__()

        self.out = nn.Sequential(
            conv2d(256, 64, 1),
            nn.Conv2d(64, 1, 1)
        )

    def forward(self, x, pred):
        residual = x
        # residual1 = self.out(residual)
        # residual_out = torch.squeeze(residual1).cpu().numpy()  # 320,320
        # print(residual_out)
        # io.imsave(f"./aaa/{opt.model}/old4.jpg", residual_out)

        score = torch.sigmoid(pred)
        # score_out = torch.squeeze(score).cpu().numpy()  # 320,320
        # print(score_out)
        # io.imsave(f"./aaa/{opt.model}/predict4.jpg", score_out)

        dist = torch.abs(score - 0.5)
        att = 1 - (dist / 0.5)
        # att1_out = torch.squeeze(att).cpu().numpy()  # 320,320
        # print(att1_out)
        # io.imsave(f"./aaa/{opt.model}/lca4.jpg", att1_out)



        att_x = x * att
        # att_lca1 = self.out(att_x)
        # # att_lca1 = torch.sigmoid(att_lca1)
        # att_lca1_out = torch.squeeze(att_lca1).cpu().numpy()  # 320,320
        # print(att_lca1_out)
        # io.imsave(f"./aaa/{opt.model}/lca_out4.jpg", att_lca1_out)



        out = att_x + residual

        # out1 = self.out(out)
        # # att_lca1 = torch.sigmoid(att_lca1)
        # out1 = torch.squeeze(out1).cpu().numpy()  # 320,320
        # print(out1)
        # io.imsave(f"./aaa/{opt.model}/lca_out_residual4.jpg", out1)


        return out


""" Global Context Module"""

class GCM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCM, self).__init__()
        pool_size = [1, 3, 5]
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

    def forward(self, x):
        xsize = x.size()[2:]
        global_context = []     # 0 1 2
        for i in range(len(self.GCmodule) - 1):     # 0 1 2
            global_context.append(F.interpolate(self.GCmodule[i](x), xsize, mode='bilinear', align_corners=True))
        global_context.append(self.GCmodule[-1](x))     # 0 1 2 3
        global_context = torch.cat(global_context, dim=1)

        output = []
        for i in range(len(self.GCoutmodel)):       # 0 1 2 3
            output.append(self.GCoutmodel[i](global_context))

        return output



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