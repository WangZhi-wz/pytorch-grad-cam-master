import torch
import torch.nn as nn
import torchvision.models as models

from models.ACSNet_DCR.modules import conv2d, conv1d, PAM_Module
from models.ACSNet_DCR.modules_DCR import LCA, GCM, ASM, NonLocalBlock


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

        self.dropout = nn.Dropout2d(0.5)

        self.conv2 = nn.Conv2d(in_channels // 4, out_channels, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.conv2(x)

        return x


class ACSNet_DCR(nn.Module):
    def __init__(self,
                 bank_size=20,
                 num_classes=1,
                 num_channels=3,
                 is_deconv=False,
                 decoder_kernel_size=3,
                 pretrained=True,
                 feat_channels=512
                 ):
        super(ACSNet_DCR, self).__init__()

        self.bank_size = 20
        self.register_buffer("bank_ptr", torch.zeros(1, dtype=torch.long))  # memory bank pointer
        self.register_buffer("bank", torch.zeros(self.bank_size, feat_channels, num_classes))  # memory bank
        self.bank_full = False

        # =====Attentive Cross Image Interaction==== #
        self.feat_channels = feat_channels
        self.L = nn.Conv2d(feat_channels, num_classes, 1)
        self.X = conv2d(feat_channels, 512, 3)
        self.phi = conv1d(512, 256)
        self.psi = conv1d(512, 256)
        self.delta = conv1d(512, 256)
        self.rho = conv1d(256, 512)
        self.g = conv2d(512 + 512, 512, 1)

        # =========Dual Attention========== #
        self.sa_head = PAM_Module(feat_channels)

        # =========Attention Fusion=========#
        self.fusion = nn.Conv2d(feat_channels, feat_channels, 1)


        GCoutlist = []
        out_channel_list = [256, 128, 64, 64]
        upsampe_scale = [2, 4, 8, 16]
        for i in range(4):      # 0 1 2 3
            GCoutlist.append(nn.Sequential(nn.Conv2d(512, out_channel_list[i], 3, 1, 1),
                                           nn.ReLU(inplace=True),
                                           nn.Upsample(scale_factor=upsampe_scale[i], mode='bilinear')))
        self.GCoutmodel = nn.ModuleList(GCoutlist)

        self.nonconv = nn.Sequential(
            nn.Conv2d(512, 512, 1, 1),
            nn.ReLU(inplace=True),
            NonLocalBlock(512))


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
        self.lca1 = LCA()
        self.lca2 = LCA()
        self.lca3 = LCA()
        self.lca4 = LCA()

        # global context module
        self.gcm = GCM(512, 64)

        # adaptive selection module
        self.asm4 = ASM(512, 1024)
        self.asm3 = ASM(256, 512)
        self.asm2 = ASM(128, 256)
        self.asm1 = ASM(64, 192)

    #==Initiate the pointer of bank buffer==#
    def init(self):
        self.bank_ptr[0] = 0
        self.bank_full = False

    @torch.no_grad()  # 这句很重要！！！！
    def update_bank(self, x):
        ptr = int(self.bank_ptr)
        batch_size = x.shape[0]
        vacancy = self.bank_size - ptr  # 空缺
        if batch_size >= vacancy:
            self.bank_full = True
        pos = min(batch_size, vacancy)
        self.bank[ptr:ptr + pos] = x[0:pos].clone()
        # update pointer
        ptr = (ptr + pos) % self.bank_size
        self.bank_ptr[0] = ptr

    def region_representation(self, input):
        X = self.X(input)
        L = self.L(input)
        aux_out = L
        batch, n_class, height, width = L.shape
        l_flat = L.view(batch, n_class, -1)
        # M = B * N * HW
        M = torch.softmax(l_flat, -1)
        channel = X.shape[1]
        # X_flat = B * C * HW
        X_flat = X.view(batch, channel, -1)
        # f_k = B * C * N
        f_k = (M @ X_flat.transpose(1, 2)).transpose(1, 2)
        return aux_out, f_k, X_flat, X

    def attentive_interaction(self, bank, X_flat, X):
        batch, n_class, height, width = X.shape
        # query = S * C
        query = self.phi(bank).squeeze(dim=2)
        # key: = B * C * HW
        key = self.psi(X_flat)
        # logit = HW * S * B (cross image relation)
        logit = torch.matmul(query, key).transpose(0, 2)
        # attn = HW * S * B
        attn = torch.softmax(logit, 2)  ##softmax维度要正确

        # delta = S * C
        delta = self.delta(bank).squeeze(dim=2)
        # attn_sum = B * C * HW
        attn_sum = torch.matmul(attn.transpose(1, 2), delta).transpose(1, 2)
        # x_obj = B * C * H * W
        X_obj = self.rho(attn_sum).view(batch, -1, height, width)

        concat = torch.cat([X, X_obj], 1)
        out = self.g(concat)
        return out

    def forward(self, x, flag):
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
        e5 = self.encoder5(e4)  # 7

        # GCM
        # global_contexts = self.gcm(e5)

        # DCR
        # === Attentive Cross Image Interaction === 外部注意力#
        aux_out, patch, feats_flat, feats = self.region_representation(e5)
        if flag == 'train':
            # print("flag", flag)
            self.update_bank(patch)
            ptr = int(self.bank_ptr)
            if self.bank_full == True:
                feature_aug = self.attentive_interaction(self.bank, feats_flat, feats)
            else:
                feature_aug = self.attentive_interaction(self.bank[0:ptr], feats_flat, feats)
        elif flag == 'test':
            # print("flag", flag)
            feature_aug = self.attentive_interaction(patch, feats_flat, feats)

        # === Dual Attention === 内部注意力#
        sa_feat = self.sa_head(e5)

        # === Fusion === 将外部注意和内部注意相加#
        feat = sa_feat + feature_aug  # (4, 512, 8, 8)

        # Non-local #
        nonfeature = self.nonconv(e5)

        feats = nonfeature + feat

        output = []
        for i in range(len(self.GCoutmodel)):  # 0 1 2 3
            output.append(self.GCoutmodel[i](feats))


        # Decoder5
        d5 = self.decoder5(e5)  # 14
        out5 = self.sideout5(d5)
        lc4  = self.lca4(e4, out5)
        gc4 = output[0]
        comb4 = self.asm4(lc4, d5, gc4)

        # Decoder4
        d4 = self.decoder4(comb4)  # 28
        out4 = self.sideout4(d4)
        lc3 = self.lca3(e3, out4)
        gc3 = output[1]
        comb3 = self.asm3(lc3, d4, gc3)

        # Decoder3
        d3 = self.decoder3(comb3)  # 56
        out3 = self.sideout3(d3)
        lc2 = self.lca2(e2, out3)
        gc2 = output[2]
        comb2 = self.asm2(lc2, d3, gc2)

        # Decoder2
        d2 = self.decoder2(comb2)  # 128
        out2 = self.sideout2(d2)
        lc1 = self.lca1(e1, out2)
        gc1 = output[3]
        comb1 = self.asm1(lc1, d2, gc1)

        # Decoder1
        d1 = self.decoder1(comb1)  # 224*224*64
        out1 = self.outconv(d1)  # 224

        return torch.sigmoid(out1), torch.sigmoid(out2), torch.sigmoid(out3), \
            torch.sigmoid(out4), torch.sigmoid(out5)
