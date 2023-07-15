import torch
from torch import nn
from torch.nn import functional as F



class text_classifier(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Sequential(
            nn.Linear(in_c, in_c//8, bias=False), nn.ReLU(),
            nn.Linear(in_c//8, out_c[0], bias=False)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_c, in_c//8, bias=False), nn.ReLU(),
            nn.Linear(in_c//8, out_c[1], bias=False)
        )

    def forward(self, feats):
        pool = self.avg_pool(feats).view(feats.shape[0], feats.shape[1])
        num_polyps = self.fc1(pool)
        polyp_sizes = self.fc2(pool)
        return num_polyps, polyp_sizes


class embedding_feature_fusion(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Conv2d((in_c[0]+in_c[1])*in_c[2], out_c, 1, bias=False), nn.ReLU(),
            nn.Conv2d(out_c, out_c, 1, bias=False), nn.ReLU()
        )

    def forward(self, num_polyps, polyp_sizes, label):
        num_polyps_prob = torch.softmax(num_polyps, axis=1)
        polyp_sizes_prob = torch.softmax(polyp_sizes, axis=1)
        prob = torch.cat([num_polyps_prob, polyp_sizes_prob], axis=1)
        prob = prob.view(prob.shape[0], prob.shape[1], 1)
        x = label * prob
        x = x.view(x.shape[0], -1, 1, 1)
        x = self.fc(x)
        x = x.view(x.shape[0], -1)
        return x




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
            GClist.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(ps),
                nn.Conv2d(in_channels, out_channels, 1, 1),
                nn.ReLU(inplace=True)))
        GClist.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1),
            nn.ReLU(inplace=True),
            NonLocalBlock(out_channels)))
        self.GCmodule = nn.ModuleList(GClist)
        for i in range(4):
            GCoutlist.append(nn.Sequential(nn.Conv2d(out_channels * 4, out_channel_list[i], 3, 1, 1),
                                           nn.ReLU(inplace=True),
                                           nn.Upsample(scale_factor=upsampe_scale[i], mode='bilinear')))
        self.GCoutmodel = nn.ModuleList(GCoutlist)

    def forward(self, x):
        xsize = x.size()[2:]
        global_context = []
        for i in range(len(self.GCmodule) - 1):
            global_context.append(F.interpolate(self.GCmodule[i](x), xsize, mode='bilinear', align_corners=True))
        global_context.append(self.GCmodule[-1](x))
        global_context = torch.cat(global_context, dim=1)

        output = []
        for i in range(len(self.GCoutmodel)):
            output.append(self.GCoutmodel[i](global_context))

        return output


""" Adaptive Selection Module"""


class ASM(nn.Module):
    def __init__(self, in_channels, all_channels, lc_channel):
        super(ASM, self).__init__()
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