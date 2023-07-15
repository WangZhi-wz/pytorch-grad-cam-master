import torch
import torch.nn as nn
import torchvision.models as models

from models.ACSNet_CSAM_LLCA.modules_CSAM_LLCA import CBAMBlock
from models.ACSNet_LLCA.modules_LLCA import LCA, GCM, ASM, LCA_modify, LCA_modify_end


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=1):
        super(DecoderBlock, self).__init__()

        self.conv1 = ConvBlock(in_channels, in_channels // 4, kernel_size=kernel_size,
                               stride=stride, padding=padding)

        self.conv2 = ConvBlock(in_channels // 4, out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.upsample(x)
        return x


class SideoutBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(SideoutBlock, self).__init__()

        self.conv1 = ConvBlock(in_channels, in_channels // 4, kernel_size=kernel_size,
                               stride=stride, padding=padding)

        self.dropout = nn.Dropout2d(0.1)

        self.conv2 = nn.Conv2d(in_channels // 4, out_channels, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.conv2(x)

        return x


class ACSNet_CSAM_LLCA(nn.Module):
    def __init__(self, num_classes):
        super(ACSNet_CSAM_LLCA, self).__init__()

        resnet = models.resnet34(pretrained=True)
        
        # Encoder
        self.encoder1_conv = resnet.conv1
        self.encoder1_bn = resnet.bn1
        self.encoder1_relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.encoder2 = resnet.layer1
        self.encoder3 = resnet.layer2
        self.encoder4 = resnet.layer3
        self.encoder5 = resnet.layer4

        # cbam attention
        self.cbam = CBAMBlock(channel=512, reduction=16)

        # Decoder
        self.decoder5 = DecoderBlock(in_channels=512, out_channels=512)
        self.decoder4 = DecoderBlock(in_channels=1024, out_channels=256)
        self.decoder3 = DecoderBlock(in_channels=512, out_channels=128)
        self.decoder2 = DecoderBlock(in_channels=256, out_channels=64)
        self.decoder1 = DecoderBlock(in_channels=192, out_channels=64)
        self.outconv = nn.Sequential(ConvBlock(64, 32, kernel_size=3, stride=1, padding=1),
                                      nn.Dropout2d(0.1),
                                      nn.Conv2d(32, num_classes, 1))

        # Sideout
        self.sideout2 = SideoutBlock(64, 1)
        self.sideout3 = SideoutBlock(128, 1)
        self.sideout4 = SideoutBlock(256, 1)
        self.sideout5 = SideoutBlock(512, 1)



        # local context attention module
        self.lca4 = LCA()
        self.lca3 = LCA_modify(256)
        self.lca2 = LCA_modify(128)
        self.lca1 = LCA_modify_end()


        # global context module  GCM -> AGCM
        self.gcm = GCM(512, 64)


        # adaptive selection module
        self.asm4 = ASM(512, 1024)
        self.asm3 = ASM(256, 512)
        self.asm2 = ASM(128, 256)
        self.asm1 = ASM(64, 192)

    def forward(self, x):
        # x 224
        # Encoder1
        e1 = self.encoder1_conv(x)      # (1, 64, 128, 128)
        e1 = self.encoder1_bn(e1)       # (1, 64, 128, 128)
        e1 = self.encoder1_relu(e1)     # (1, 64, 128, 128)
        e1_pool = self.maxpool(e1)      # (1, 64, 64, 64)
        # Encoder2
        e2 = self.encoder2(e1_pool)     # conv(64, stride=1) -> (1, 64, 64, 64)
        # Encoder3
        e3 = self.encoder3(e2)          # conv(128, stride=2) -> (1, 128, 32, 32)
        # Encoder4
        e4 = self.encoder4(e3)          # conv(256, stride=2) -> (1, 256, 16, 16)
        # Encoder5
        e5 = self.encoder5(e4)          # conv(512, stride=2) -> (1, 512, 8, 8)

        # SA attention  e5 -> d5
        e5_att = self.cbam(e5)

        # GCM
        global_contexts = self.gcm(e5_att)  # 0:(1, 256, 16, 16) 1:(1, 128, 32, 32) 2:(1, 64, 64, 64) 3:(1, 64, 128, 128)


        # Decoder5
        d5 = self.decoder5(e5_att)          # conv1(512,128)->conv2(128,512)->upsample-> (1, 512, 16, 16)
        out5 = self.sideout5(d5)        # conv1(512,128)->drop(0.1)->conv2(128,1)-> (2, 1, 16, 16)
        lc4 = self.lca4(e4, out5)       # e4:256 x 24 x 18 + out5:1 x 24 x 18 -> (2, 256, 16, 16)
        gc4 = global_contexts[0]        # 0:(2, 256, 16, 16)
        comb4 = self.asm4(lc4, d5, gc4)
        # (2, 256, 16, 16)、(1, 512, 16, 16)、(2, 256, 16, 16) = (2, 1024, 16, 16)


        # Decoder4
        d4 = self.decoder4(comb4)       # conv1(1024,256)->conv2(256,256)->upsample-> (2, 256, 32, 32)
        out4 = self.sideout4(d4)        # conv1(256,64)->drop(0.1)->conv2(64,1)-> (2, 1, 32, 32)
        lc3 = self.lca3(e3, out4, lc4)       # e3:(1, 128, 32, 32) + out5:(2, 1, 32, 32) -> (2, 128, 32, 32)
        gc3 = global_contexts[1]        # 1:(2, 128, 32, 32)
        comb3 = self.asm3(lc3, d4, gc3)
        # (2, 128, 32, 32)、(2, 256, 32, 32)、(2, 128, 32, 32) = (2, 512 32, 32)


        # Decoder3
        d3 = self.decoder3(comb3)       # conv1(512,128)->conv2(128,128)->upsample-> (2, 128, 64, 64)
        out3 = self.sideout3(d3)        # conv1(128,32)->drop(0.1)->conv2(32,1)-> (2, 1, 64, 64)
        lc2 = self.lca2(e2, out3, lc3)       # e2:(1, 64, 64, 64) + out5:(2, 1, 64, 64) -> (2, 64, 64, 64)
        gc2 = global_contexts[2]        # 2:(2, 64, 64, 64)
        comb2 = self.asm2(lc2, d3, gc2)
        # (2, 64, 64, 64)、(2, 128, 64, 64)、(2, 64, 64, 64) = (2, 256, 64, 64)


        # Decoder2
        d2 = self.decoder2(comb2)       # conv1(256,64)->conv2(64,64)->upsample-> (2, 64, 128, 128)
        out2 = self.sideout2(d2)        # conv1(64,16)->drop(0.1)->conv2(16,1)-> (2, 1, 128, 128)
        lc1 = self.lca1(e1, out2, lc2)       # e1:(1, 64, 128, 128)  + out5:(2, 1, 128, 128) -> (2, 64, 128, 128)
        gc1 = global_contexts[3]        # 3：(2, 64, 128, 128)
        comb1 = self.asm1(lc1, d2, gc1)
        # (2, 64, 128, 128)、(2, 64, 128, 128)、(2, 64, 128, 128) = (2, 192, 128, 128)


        # Decoder1
        d1 = self.decoder1(comb1)       # (2, 64, 256, 256)
        out1 = self.outconv(d1)         # (2, 1, 256, 256)


        return torch.sigmoid(out1), torch.sigmoid(out2), torch.sigmoid(out3), \
               torch.sigmoid(out4), torch.sigmoid(out5)
