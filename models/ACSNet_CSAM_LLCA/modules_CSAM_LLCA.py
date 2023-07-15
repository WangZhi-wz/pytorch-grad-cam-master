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

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output

""" CBAMBlock """
class CBAMBlock(nn.Module):

    def __init__(self, channel=512,reduction=16,kernel_size=49):
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




""" Local Context Attention Module"""

# 局部上下文注意： 关注分值不确定的区域，更加关注预测图越靠近0.5的区域，减少趋近0和1的关注-----------------------------4
class LCA(nn.Module):
    def __init__(self):
        super(LCA, self).__init__()

    def forward(self, x, pred):
        residual = x
        score = torch.sigmoid(pred)
        dist = torch.abs(score - 0.5)
        att = 1 - (dist / 0.5)

        # att_max = torch.max(att)
        # att_min = torch.min(att)
        # att_map = (att_map-att_min)/(att_max-att_min)
        # print(att_map)
        att_x = x * att

        out = att_x + residual

        return out



# 局部上下文注意： 关注分值不确定的区域，更加关注预测图越靠近0.5的区域，减少趋近0和1的关注-----------------------------3 2
class LCA_modify(nn.Module):
    def __init__(self, in_channels):
        super(LCA_modify, self).__init__()
        self.convl = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, stride=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    #
    def forward(self, x, pred, lca_down):
        residual = x
        score = torch.sigmoid(pred)
        dist = torch.abs(score - 0.5)
        att = 1 - (dist / 0.5)
        att_x = x * att

        lca_down = self.convl(lca_down)
        lca_down = self.upsample(lca_down)

        out = att_x + residual + lca_down

        return out



# 局部上下文注意： 关注分值不确定的区域，更加关注预测图越靠近0.5的区域，减少趋近0和1的关注----------------------------- 1
class LCA_modify_end(nn.Module):
    def __init__(self):
        super(LCA_modify_end, self).__init__()
        # self.convl = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, stride=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x, pred, lca_down):
        residual = x
        score = torch.sigmoid(pred)
        dist = torch.abs(score - 0.5)
        att = 1 - (dist / 0.5)
        att_x = x * att

        # lca_down = self.convl(lca_down)
        lca_down = self.upsample(lca_down)

        out = att_x + residual + lca_down

        return out



""" Global Context Module"""

# GCM(512, 64)
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
                nn.AdaptiveAvgPool2d(ps),   # 1 3 5
                nn.Conv2d(in_channels, out_channels, 1, 1),
                nn.ReLU(inplace=True)))

        GClist.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1),
            nn.ReLU(inplace=True),
            NonLocalBlock(out_channels)))
        self.GCmodule = nn.ModuleList(GClist)   # GClist[0] GClist[1] GClist[2]

        for i in range(4):
            GCoutlist.append(nn.Sequential(
                nn.Conv2d(out_channels * 4, out_channel_list[i], 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=upsampe_scale[i], mode='bilinear')))
        self.GCoutmodel = nn.ModuleList(GCoutlist)  # GCoutlist[0] GCoutlist[1] GCoutlist[2] GCoutlist[3]


    def forward(self, x):   #(2, 512, 12, 9)
        xsize = x.size()[2:]    # 输入图片尺寸 (12, 9)
        global_context = []     # 0:(2, 64, 12, 9)  1:0:(2, 64, 12, 9)  2:0:(2, 64, 12, 9)
        for i in range(len(self.GCmodule) - 1):
            global_context.append(F.interpolate(self.GCmodule[i](x), xsize, mode='bilinear', align_corners=True))
        global_context.append(self.GCmodule[-1](x))      # 0:(2, 64, 12, 9)  1:(2, 64, 12, 9)  2:(2, 64, 12, 9)  3:(2, 64, 12, 9)
        global_context = torch.cat(global_context, dim=1)   # (2, 256, 12, 9)

        output = []
        for i in range(len(self.GCoutmodel)):
            output.append(self.GCoutmodel[i](global_context))   # (2, 256, 24, 18)  (2, 128, 48, 36)  (2, 64, 96, 72)  (2, 64, 192, 144)

        return output




""" Adaptive Selection Module"""


class ASM(nn.Module):
    def __init__(self, in_channels, all_channels):
        super(ASM, self).__init__()
        self.non_local = NonLocalBlock(in_channels)
        self.selayer = SELayer(all_channels)

    def forward(self, lc, fuse, gc):

        fuse = self.non_local(fuse)
        fuse = torch.cat([lc, fuse, gc], dim=1) # (2, 1024, 24, 18)
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

        batch_size = x.size(0)  # 2

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)   # x:(2, 64 ,12 ,9)->(2, 64 ,24 ,18)
        g_x = g_x.permute(0, 2, 1)  # g_x(2, 24, 32)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)   # (2, 32, 108)
        theta_x = theta_x.permute(0, 2, 1)  # (2, 108, 32)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)   # (2, 32, 24)
        f = torch.matmul(theta_x, phi_x)    # (2, 108, 24)
        f_div_C = F.softmax(f, dim=-1)  # (2, 108, 24)

        y = torch.matmul(f_div_C, g_x)  # (2, 108, 32)
        y = y.permute(0, 2, 1).contiguous() # (2, 32, 108)
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])  # (2, 32, 12 ,9)
        W_y = self.W(y) # (2, 64, 12 ,9)
        z = W_y + x # (2, 64, 12 ,9)

        return z




# # 局部上下文注意： 关注分值不确定的区域，更加关注预测图越靠近0.5的区域，减少趋近0和1的关注
# class LCA_modify1(nn.Module):
#     def __init__(self):
#         super(LCA_modify1, self).__init__()
#
#     def forward(self, x, pred, pred_up):
#         residual = x
#         score = torch.sigmoid(pred)
#         dist = torch.abs(score - 0.5)
#         att = 1 - (dist / 0.5)
#         att_x1 = x * att
#
#         score_up = torch.sigmoid(pred_up)
#         dist_up = torch.abs(score_up - 0.5)
#         att_up = 1- (dist_up / 0.5)
#         att_x2 = x * att_up
#
#         out = att_x1 + att_x2 + residual
#
#         return out
#
# # 局部上下文注意： 关注分值不确定的区域，更加关注预测图越靠近0.5的区域，减少趋近0和1的关注
# class LCA_modify2(nn.Module):
#     def __init__(self):
#         super(LCA_modify2, self).__init__()
#
#     def forward(self, x, pred, pred_up):
#         residual = x
#         score = torch.sigmoid(pred)
#         dist = torch.abs(score - 0.5)
#         att = 1 - (dist / 0.5)
#         att_x1 = x * att
#
#         score_up = torch.sigmoid(pred_up)
#         dist_up = torch.abs(score_up - 0.5)
#         att_up = 1- (dist_up / 0.5)
#         att_x2 = x * att_up
#
#         out = att_x1 + att_x2 + residual
#
#         return out
#
#
# # 局部上下文注意： 关注分值不确定的区域，更加关注预测图越靠近0.5的区域，减少趋近0和1的关注
# class LCA_modify3(nn.Module):
#     def __init__(self):
#         super(LCA_modify3, self).__init__()
#
#     def forward(self, x, pred, pred_down):
#         residual = x
#         score = torch.sigmoid(pred)
#         dist = torch.abs(score - 0.5)
#         att = 1 - (dist / 0.5)
#         att_x1 = x * att
#
#         score_down = torch.sigmoid(pred_down)
#         dist_down = torch.abs(score_down - 0.5)
#         att_down = 1- (dist_down / 0.5)
#         att_x2 = x * att_down
#
#         out = att_x1 + att_x2 + residual
#
#         return out