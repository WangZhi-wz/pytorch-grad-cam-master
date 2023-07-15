import torch
import torch.nn as nn
import torchvision.models as models

from models.ACSNet_CSAM_in_LLCA.modules_CSAM_in_LLCA import LCA, GCM, ASM, CSAMBlock,LCA_modify,LCA_modify_end


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


class ACSNet_CSAM_in_LLCA(nn.Module):
    def __init__(self, num_classes):
        super(ACSNet_CSAM_in_LLCA, self).__init__()

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
        self.csam1 = CSAMBlock(channel=64, reduction=16)
        self.csam2 = CSAMBlock(channel=64, reduction=16)
        self.csam3 = CSAMBlock(channel=128, reduction=16)
        self.csam4 = CSAMBlock(channel=256, reduction=16)
        self.csam5 = CSAMBlock(channel=512, reduction=16)

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

        # global context module
        self.gcm = GCM(512, 64)

        # adaptive selection module
        self.asm4 = ASM(512, 1024)
        self.asm3 = ASM(256, 512)
        self.asm2 = ASM(128, 256)
        self.asm1 = ASM(64, 192)

    def forward(self, x):
        # x 224
        # Encoder1
        e1 = self.encoder1_conv(x)  # 128
        e1 = self.encoder1_bn(e1)
        e1 = self.encoder1_relu(e1)
        e1_pool = self.maxpool(e1)  # 56
        # Encoder2
        e2 = self.encoder2(e1_pool)
        # Encoder3
        e3 = self.encoder3(e2)  # 28
        # Encoder4
        e4 = self.encoder4(e3)  # 14
        # Encoder5
        e5 = self.encoder5(e4)  # conv(512, stride=2) -> (1, 512, 8, 8)


        # SA attention
        e1_att = self.csam1(e1)
        e2_att = self.csam2(e2)
        e3_att = self.csam3(e3)
        e4_att = self.csam4(e4)
        e5_att = self.csam5(e5)

        # GCM
        global_contexts = self.gcm(e5)

        # Decoder5
        d5 = self.decoder5(e5_att)  # 14
        out5 = self.sideout5(d5)
        lc4 = self.lca4(e4_att, out5)
        gc4 = global_contexts[0]
        comb4 = self.asm4(lc4, d5, gc4)

        # Decoder4
        d4 = self.decoder4(comb4)  # 28
        out4 = self.sideout4(d4)
        lc3 = self.lca3(e3_att, out4, lc4)
        gc3 = global_contexts[1]
        comb3 = self.asm3(lc3, d4, gc3)

        # Decoder3
        d3 = self.decoder3(comb3)  # 56
        out3 = self.sideout3(d3)
        lc2 = self.lca2(e2_att, out3, lc3)
        gc2 = global_contexts[2]
        comb2 = self.asm2(lc2, d3, gc2)

        # Decoder2
        d2 = self.decoder2(comb2)  # 128
        out2 = self.sideout2(d2)
        lc1 = self.lca1(e1_att, out2, lc2)
        gc1 = global_contexts[3]
        comb1 = self.asm1(lc1, d2, gc1)

        # Decoder1
        d1 = self.decoder1(comb1)  # 224*224*64
        out1 = self.outconv(d1)  # 224

        return torch.sigmoid(out1), torch.sigmoid(out2), torch.sigmoid(out3), \
            torch.sigmoid(out4), torch.sigmoid(out5)
