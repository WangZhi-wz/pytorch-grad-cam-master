import torch
from torch import nn
from torch.nn import functional as F

from models.ACSNet_UACA.layers import conv

""" Local Context Attention Module"""

class LCA(nn.Module):
    def __init__(self):
        super(LCA, self).__init__()

    def forward(self, x, pred):
        residual = x
        score = torch.sigmoid(pred)
        dist = torch.abs(score - 0.5)
        att = 1 - (dist / 0.5)

        att_x = x * att

        out = att_x + residual

        return out



""" UACA"""

class UACA(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UACA, self).__init__()
        self.channel = out_channels

        self.conv_query = nn.Sequential(conv(in_channels, out_channels, 3, relu=True),
                                        conv(out_channels, out_channels, 3, relu=True))
        self.conv_key = nn.Sequential(conv(in_channels, out_channels, 1, relu=True),
                                      conv(out_channels, out_channels, 1, relu=True))
        self.conv_value = nn.Sequential(conv(in_channels, out_channels, 1, relu=True),
                                        conv(out_channels, out_channels, 1, relu=True))

        self.conv_out1 = conv(out_channels, 1, 3, relu=True)
        self.conv_out2 = conv(in_channels + out_channels, out_channels, 3, relu=True)
        self.conv_out3 = conv(out_channels, out_channels, 3, relu=True)
        self.conv_out4 = conv(out_channels, 1, 1)



    def forward(self, x, map):      # x(2, 256, 16, 16)  map(2, 1, 16, 16)
        residual = x
        b, c, h, w = x.shape        # b 2 c 256 h 16 w 16
        # compute class probability
        map = F.interpolate(map, size=x.shape[-2:], mode='bilinear', align_corners=False)   # (2, 1, 16, 16)
        fg = torch.sigmoid(map)     # (2, 1, 16, 16)

        p = fg - .5     # (2, 1, 16, 16)

        fg = torch.clip(p, 0, 1)  # foreground (2, 1, 16, 16)
        bg = torch.clip(-p, 0, 1)  # background (2, 1, 16, 16)
        cg = .5 - torch.abs(p)  # confusion area (2, 1, 16, 16)

        prob = torch.cat([fg, bg, cg], dim=1)   # (2, 3, 16, 16)

        # reshape feature & prob
        f = x.view(b, h * w, -1)    # (2, 256, 256)
        prob = prob.view(b, 3, h * w)   # (2, 3, 256)

        # compute context vector 计算上下文向量
        context = torch.bmm(prob, f).permute(0, 2, 1).unsqueeze(3)  # b, 3, c (2, 256, 3, 1)

        # k q v compute
        query = self.conv_query(x).view(b, self.channel, -1).permute(0, 2, 1)   # (2, 256, 64)
        key = self.conv_key(context).view(b, self.channel, -1)      # (2, 64, 3)
        value = self.conv_value(context).view(b, self.channel, -1).permute(0, 2, 1)     # (2, 3, 64)

        # compute similarity map 计算相似度图
        sim = torch.bmm(query, key)  # b, hw, c x b, c, 2   (2, 256, 3)
        sim = (self.channel ** -.5) * sim   # (2, 256, 3)
        sim = F.softmax(sim, dim=-1)    # (2, 256, 3)

        # compute refined feature 计算优化特征
        context = torch.bmm(sim, value).permute(0, 2, 1).contiguous().view(b, -1, h, w)     # (2, 64, 16, 16)
        context = self.conv_out1(context)   # (2, 64, 16, 16)


        # x = torch.cat([x, context], dim=1)     # (2, 320, 16, 16)
        # x = self.conv_out2(x)     # (2, 64, 16, 16)
        # x = self.conv_out3(x)     # (2, 64, 16, 16)
        # out = self.conv_out4(x)
        # out = out + map

        # return out
        x = x * context
        return x
        # attx = residual * context
        # out = residual + attx
        # return out

        # residual = x
        # score = torch.sigmoid(pred)
        # dist = torch.abs(score - 0.5)
        # att = 1 - (dist / 0.5)
        #
        # att_x = x * att
        #
        # out = att_x + residual
        #
        # return out



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