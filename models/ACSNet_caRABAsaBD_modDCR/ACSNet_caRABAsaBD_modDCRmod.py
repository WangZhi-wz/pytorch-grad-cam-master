import torch
from torch.nn import functional as F
import torch.nn as nn
import torchvision.models as models
from models.ACSNet_caRABAsaBD_modDCR.modules_caRABAsaBD_modDCR import GCM, ASM, GCM_RFB, ASM_nogcm, CSAMBlock, \
    LCA11, LCA32, LCA44, RA1, RA4, NonLocalBlock, CSAMBlock1, CSAMBlock432, GCM_RFB_s, ChannelAttention, SpatialAttention, SELayer,ChannelAttention_mod, ChannelAttention_modeca, ChannelAttention_gam, ChannelAttention_gam1, ChannelAttention_gam12, \
    ChannelAttention_gam123, RA3, RA2, SKAttention
from models.ACSNet_caRABAsaBD_modDCR.modules import conv2d, conv1d, PAM_Module, CAM_Module



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

class convxiajiang(nn.Module):
    def __init__(self, in_channels):
        super(convxiajiang, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels*2,
                              kernel_size=3,
                              stride=1,
                              padding=1)
        self.bn = nn.BatchNorm2d(in_channels*2)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, ceil_mode=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.maxpool(x)
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


class ACSNet_caRABAsaBD_modDCRmod(nn.Module):
    def __init__(self,
                 bank_size=20,
                 num_classes=1,
                 num_channels=3,
                 is_deconv=False,
                 decoder_kernel_size=3,
                 pretrained=True,
                 feat_channels = 512
                 ):
        super(ACSNet_caRABAsaBD_modDCRmod, self).__init__()

        self.bank_size = 20
        self.register_buffer("bank_ptr", torch.zeros(1, dtype=torch.long))  # memory bank pointer
        self.register_buffer("bank", torch.zeros(self.bank_size, feat_channels, num_classes))  # memory bank
        self.register_buffer("bank_ptr1", torch.zeros(1, dtype=torch.long))  # memory bank pointer
        self.register_buffer("bank1", torch.zeros(self.bank_size, 64, 1))  # memory bank
        self.bank_full = False
        self.bank_full1 = False

        # =====Attentive Cross Image Interaction==== #
        self.feat_channels = feat_channels
        self.L = nn.Conv2d(feat_channels, num_classes, 1)
        self.X = conv2d(feat_channels, 512, 3)
        self.phi = conv1d(512, 256)
        self.psi = conv1d(512, 256)
        self.delta = conv1d(512, 256)
        self.rho = conv1d(256, 512)
        self.g = conv2d(512 + 512, 512, 1)
        self.ska = SKAttention(512)
        self.ca = ChannelAttention(512)    # 修改-------------------------------------------------------------
        self.ca_scSE = ChannelAttention_mod(512)   # 修改-------------------------------------------------------------
        self.ca_eca = ChannelAttention_modeca(512)
        self.ca_gam = ChannelAttention_gam(512)
        self.ca_gam1 = ChannelAttention_gam1(512)
        self.casa_gam = ChannelAttention_gam12(512)
        self.casa_sSE = ChannelAttention_gam123(512)
        self.sa = SpatialAttention()    # 修改-------------------------------------------------------------
        self.maxpoolaa = nn.AdaptiveMaxPool2d(1)  # 修改-------------------------------------------------------------
        self.avgpoolaa = nn.AdaptiveAvgPool2d(1)  # 修改-------------------------------------------------------------
        self.boundary = nn.Sequential(nn.Conv2d(512, 128, kernel_size=1, stride=1),
                                       nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                                       nn.Conv2d(128, 1, kernel_size=1, stride=1, bias=False),
                                       nn.Sigmoid())


        self.pool1 = nn.Sequential(  # Global average pool -> 3x3AdaptiveAvgPool -> 5x5AdaptiveAvgPool
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 128, 1, 1),
            nn.ReLU(inplace=True),
        )
        self.pool3 = nn.Sequential(  # Global average pool -> 3x3AdaptiveAvgPool -> 5x5AdaptiveAvgPool
            nn.AdaptiveAvgPool2d(3),
            nn.Conv2d(512, 128, 1, 1),
            nn.ReLU(inplace=True))
        self.pool5 = nn.Sequential(  # Global average pool -> 3x3AdaptiveAvgPool -> 5x5AdaptiveAvgPool
            nn.AdaptiveAvgPool2d(5),
            nn.Conv2d(512, 128, 1, 1),
            nn.ReLU(inplace=True))
        self.pool7 = nn.Sequential(            # Non-Local
            nn.Conv2d(512, 128, 1, 1),
            nn.ReLU(inplace=True),
            NonLocalBlock(128))

        self.se = SELayer(512)

        # =========Dual Attention========== #
        # self.sa_head1 = PAM_Module(feat_channels)
        # self.sa_head2 = CAM_Module(feat_channels)
        # self.sa_head = CSAMBlock()
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



        resnet = models.resnet34(pretrained=True)

        # Encoder
        self.encoder1_conv = resnet.conv1
        self.encoder1_bn = resnet.bn1
        self.encoder1_relu = resnet.relu
        self.maxpool = resnet.maxpool
        # self.maxpool = SoftPooling2D(3, 2)
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
        # self.outconv_final = nn.Sequential(
        #                               nn.Dropout2d(0.1),
        #                               nn.Conv2d(32, num_classes, 1))

        # Sideout
        self.sideout2 = SideoutBlock(64, 1)
        self.sideout3 = SideoutBlock(128, 1)
        self.sideout4 = SideoutBlock(256, 1)
        self.sideout5 = SideoutBlock(512, 1)

        # local context attention module
        # GCM + CBAM ?
        # self.lca1 = GCM_CBAM1(64, 64//4)
        # self.lca2 = GCM_CBAM23(64, 64//4, 128)
        # self.lca3 = GCM_CBAM23(128, 128//4, 256)
        # self.lca4 = GCM_CBAM4(256, 256//4)
        # RA
        self.lca1 = RA1(64)
        self.lca2 = RA2(64, 128)
        self.lca3 = RA3(128, 256)
        self.lca4 = RA4(256, 512)
        # ca_LLCA_sa
        # local context attention module
        # self.lca4 = LCA44(256)
        # self.lca3 = LCA32(256, 128)
        # self.lca2 = LCA32(128, 64)
        # self.lca1 = LCA11(64)
        # csam
        # self.lca1 = CSAMBlock1(channel=64, reduction=16)
        # self.lca2 = CSAMBlock432(channel=64, reduction=16)
        # self.lca3 = CSAMBlock432(channel=128, reduction=16)
        # self.lca4 = CSAMBlock432(channel=256, reduction=16)


        # global context module
        # self.gcm = GCM(512, 128)
        # self.cbam = CSAMBlock(channel=512, reduction=16)
        # global context module + cbam
        # self.gcm = GCM(512, 128)
        # self.gcm = GCM_CBAM(512, 128)
        # self.gcm = GCM_RFB(512, 128)
        # self.gcm = GCM_RFB_s(512, 128)
        # self.nonl = NonLocalBlock(512)

        # adaptive selection module
        self.asm4 = ASM(512, 1024)
        self.asm3 = ASM(256, 512)
        self.asm2 = ASM(128, 256)
        self.asm1 = ASM(64, 192)
        # self.asm4 = ASMmod(512)
        # self.asm3 = ASMmod(256)
        # self.asm2 = ASMmod(128)
        # self.asm1 = ASMmodend(64)
        # self.asm4 = ASM_nogcm(512, 1024, 256)
        # self.asm3 = ASM_nogcm(256, 512, 128)
        # self.asm2 = ASM_nogcm(128, 256, 64)
        # self.asm1 = ASM_nogcm(64, 192, 64)

        # APF
        # self.APF = APF(32, 64, 128, 32)
        # binary attention



    #==Initiate the pointer of bank buffer==#
    def init(self):
        self.bank_ptr[0] = 0
        self.bank_ptr1[0] = 0
        self.bank_full = False
        self.bank_full1 = False

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

    @torch.no_grad()  # 这句很重要！！！！
    def update_bank1(self, x):
        ptr1 = int(self.bank_ptr1)
        batch_size = x.shape[0]
        vacancy1 = self.bank_size - ptr1  # 空缺
        if batch_size >= vacancy1:
            self.bank_full1 = True
        pos1 = min(batch_size, vacancy1)
        self.bank1[ptr1:ptr1 + pos1] = x[0:pos1].clone()
        # update pointer
        ptr1 = (ptr1 + pos1) % self.bank_size
        self.bank_ptr1[0] = ptr1

    # def region_representation(self, input):
    #     X = self.X(input)
    #     L = self.L(input)
    #     aux_out = L
    #     batch, n_class, height, width = L.shape
    #     l_flat = L.view(batch, n_class, -1)
    #     # M = B * N * HW
    #     M = torch.softmax(l_flat, -1)
    #     channel = X.shape[1]
    #     # X_flat = B * C * HW
    #     X_flat = X.view(batch, channel, -1)
    #     # f_k = B * C * N
    #     f_k = (M @ X_flat.transpose(1, 2)).transpose(1, 2)
    #     return aux_out, f_k, X_flat, X
    #
    # def attentive_interaction(self, bank, X_flat, X):
    #     batch, n_class, height, width = X.shape
    #     # query = S * C
    #     query = self.phi(bank).squeeze(dim=2)
    #     # key: = B * C * HW
    #     key = self.psi(X_flat)
    #     # logit = HW * S * B (cross image relation)
    #     logit = torch.matmul(query, key).transpose(0, 2)
    #     # attn = HW * S * B
    #     attn = torch.softmax(logit, 2)  ##softmax维度要正确
    #
    #     # delta = S * C
    #     delta = self.delta(bank).squeeze(dim=2)
    #     # attn_sum = B * C * HW
    #     attn_sum = torch.matmul(attn.transpose(1, 2), delta).transpose(1, 2)
    #     # x_obj = B * C * H * W
    #     X_obj = self.rho(attn_sum).view(batch, -1, height, width)
    #
    #     concat = torch.cat([X, X_obj], 1)
    #     out = self.g(concat)
    #     return out
    # 上下文区域嵌入

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
    def region_representation(self, input):  # e5(4,512,7,7)
        # X = self.X(input)   #4,512,8,8
        # L = self.L(input)   #4,1,8,8
        #
        # aux_out = L
        # batch, n_class, height, width = L.shape
        # l_flat = L.view(batch, n_class, -1)  # 4,1,64
        # # M = B * N * HW
        # M = torch.softmax(l_flat, -1)  # 4,1,64
        # channel = X.shape[1]
        # # X_flat = B * C * HW
        # X_flat = X.view(batch, channel, -1)  # 4,512,64
        # # f_k = B * C * N
        # f_k = (M @ X_flat.transpose(1, 2)).transpose(1, 2)  # 4,512,1
        #
        # Lmax_result = self.maxpoolaa(input)   #4,512,1,1
        # Lavg_result = self.avgpoolaa(input)   #4,512,1,1
        # L1 = Lmax_result + Lavg_result   #4,512,1,1
        # batch, n_class, height, width = L1.shape
        # l1_flat = L1.view(batch, n_class, 1)    #4,512,1
        # M1 = torch.softmax(l1_flat, -1)  #4,512,1
        # f_k1 = (M1.transpose(1, 2) @ X_flat).transpose(1, 2)  # 4,64,1
        #
        # return aux_out, f_k, f_k1, X_flat, X

        batch, n_class, height, width = input.shape
        X1 = self.pool1(input)
        X1 = F.interpolate(X1, height, mode='bilinear', align_corners=True)
        X2 = self.pool3(input)
        X2 = F.interpolate(X2, height, mode='bilinear', align_corners=True)
        X3 = self.pool5(input)
        X3 = F.interpolate(X3, height, mode='bilinear', align_corners=True)
        X4 = self.pool7(input)
        # X4 = F.interpolate(X4, 8, mode='bilinear', align_corners=True)
        X = torch.cat([X1,X2,X3,X4], dim=1)
        # X = self.X(input)
        # X = self.ska(input)

        X_old = self.X(input)   # conv3x3 (4,512,7,7)
        # X_old = X_old * self.ca(X_old)

        # X_boundary1 = self.boundary(X_old)
        # X_boundary1 = X_boundary1.repeat_interleave(int(input.shape[1]), dim=1)  # 4,64,256,256
        # X_boundary2 = self.se(X_old)  # 4,64,256,256
        # X_old = X_boundary1 + X_boundary2

        # X = self.X(input)  # conv3x3 (4,512,7,7)
        batch, channel, height, width = X_old.shape  # batch=4, channel=512, height=7, width=7
        X_flat_old = X_old.view(batch, channel, -1)  # (4,512,49)

        X_flat = X.view(batch, channel, -1)  # (4,512,49)---------------------------------------
        # X1 = X * self.ca(X)
        # X_flat1 = X1.view(batch, channel, -1)  # (4,512,49)

        # L = self.sa(X)  # (4, 1, 7, 7)  L = self.L(input)  # out_channel=1 (4,1,7,7)# 修改-------------------------------------------------------------
        L = self.L(X)
        # L = self.L(input)
        aux_out = L  # (4,1,7,7)
        batch, n_class, height, width = L.shape  # batch=4, n_class=1, height=7, width=7
        l_flat = L.view(batch, n_class, -1)  # 展平(4,1,49)
        M = torch.softmax(l_flat, -1)  # (4,1,49)------------------------------------------------

        f_k = (M @ X_flat.transpose(1, 2)).transpose(1, 2)  # 上下文区域嵌入公式 (4,512,1)

        # L1 = self.avg_pool(X) # (4,512,1,1)
        # batch, n_class, height, width = L1.shape  # batch=4, n_class=512, height=1, width=1
        # l_flat1 = L1.view(batch, n_class, -1)  # 展平(4,1,49)

        f_k1 = (f_k.transpose(1, 2) @ X_flat).transpose(1, 2)

        return aux_out, f_k, f_k1, X_flat_old, X_old

    # 注意交互
    def attentive_interaction(self, bank, bank1,  X_flat, X):

        # bank(20, 512, 1)    X_flat = feats_flat = (4, 512, 64)     X = feats = (4, 512, 8, 8)
        batch, n_class, height, width = X.shape  # batch=4, n_class=512 , height=7, width=7
        # query = S * C 缓冲区域嵌入
        query = bank.squeeze(dim=2)  # q：压缩 (20, 512, 1)->(20, 512)
        # key: = B * C * HW 扁平化特征
        key = X_flat  # (4,512,64)

        # logit = HW * S * B (cross image relation) 在它们之间执行矩阵乘法
        logit = torch.matmul(query, key).transpose(0, 2)  # (64,20,4)
        # attn = HW * S * B 应用 softmax 层来计算上下文注意力图 X ∈ RHW×B×S
        attn = torch.softmax(logit, 2)  # softmax维度要正确 (64,20,4)

        # delta = S * C
        delta = self.delta(bank).squeeze(dim=2)  # (20,256)
        # attn_sum = B * C * HW
        attn_sum = torch.matmul(attn.transpose(1, 2), delta).transpose(1, 2)  # (64,256,4)
        # x_obj = B * C * H * W
        X_obj = self.rho(attn_sum).view(batch, -1, height, width)  # (4, 512, 8, 8)




        # bank(20, 512, 1)    X_flat = feats_flat = (4, 512, 64)     X = feats = (4, 512, 8, 8)
        batch, n_class, height, width = X.shape  # batch=4, n_class=512 , height=7, width=7
        # query = S * C 缓冲区域嵌入
        query1 = bank1.squeeze(dim=2)  # q：压缩 (20, 64, 1)->(20, 64)
        # key: = B * C * HW 扁平化特征
        key1 = X_flat  # (4,512,64)

        # logit = HW * S * B (cross image relation) 在它们之间执行矩阵乘法
        logit1 = torch.matmul(query1, key1.transpose(1, 2)).transpose(0, 2)  # (512,20,4)
        # attn = HW * S * B 应用 softmax 层来计算上下文注意力图 X ∈ RHW×B×S
        attn1 = torch.softmax(logit1, 2)  # softmax维度要正确 (512,20,4)

        # delta = S * C
        delta1 = bank1.squeeze(dim=2)  # (20,64)
        # attn_sum = B * C * HW
        attn_sum1 = torch.matmul(attn1.transpose(1, 2), delta1).transpose(1, 2).transpose(1, 0)  # (64,256,4)
        # x_obj = B * C * H * W
        X_obj1 = attn_sum1.reshape(batch, 512 ,8 , 8)


        # 加的对原始输入做注意力
        # X = self.sa_head(X)
        # X = self.se(X)

        # X = X * self.ca(X)
        # X = self.ska(X)

        # batch, n_class, height, width = X.shape
        # X1 = self.pool1(X)
        # X1 = F.interpolate(X1, height, mode='bilinear', align_corners=True)
        # X2 = self.pool3(X)
        # X2 = F.interpolate(X2, height, mode='bilinear', align_corners=True)
        # X3 = self.pool5(X)
        # X3 = F.interpolate(X3, height, mode='bilinear', align_corners=True)
        # X4 = self.pool7(X)
        # Xin = torch.cat([X1,X2,X3,X4], dim=1)

        # X = self.ca_scSE(X)
        # X = self.ca_eca(X)
        # X = self.ca_gam(X)
        # X = self.ca_gam1(X)
        # X = self.casa_gam(X)
        # X = self.casa_sSE(X)

        concat = torch.cat([X_obj, X_obj1], 1)   # (4, 1024, 8, 8) 这里--------------------------------------------
        out = self.g(concat) # (4, 512, 8, 8)
        out = out + X

        return out

    def forward(self, x, flag):
        # x 224
        # Encoder1
        e1 = self.encoder1_conv(x)  # 128
        e1 = self.encoder1_bn(e1)
        e1 = self.encoder1_relu(e1) # 4,64,128,128
        e1_pool = self.maxpool(e1)  # 64
        # Encoder2
        e2 = self.encoder2(e1_pool) # 4,64,64,64
        # Encoder3
        e3 = self.encoder3(e2)  # 4,128,32,32
        # Encoder4
        e4 = self.encoder4(e3)  # 4,256,16,16
        # Encoder5
        e5 = self.encoder5(e4)  # 4,512,8,8

        # e5ska = self.ska(e5)

        # 融合e1 e2 e3 e4 e5
        # e12 = e1_pool + e2
        # e123 = e3 + self.convxia123(e12)
        # e1234 = e4 + self.convxia1234(e123)
        # e12345 = e5 + self.convxia12345(e1234)
        # GCM
        #global_contexts = self.gcm(e5)
        # nonl = self.nonl(e5)


        # global_contexts = self.cbam(e5)

        # DCR
        # === Attentive Cross Image Interaction === 外部注意力#
        aux_out, patch, patch1, feats_flat, feats = self.region_representation(e5)
        # aux_out=(4,1,7,7) , patch=(4,512,1) , feats_flat=(4,512,49) , feats=(4,512,7,7)
        if flag == 'train':
            self.update_bank(patch)  # 修改-------------------------------------------------------------
            self.update_bank1(patch1)
            ptr = int(self.bank_ptr)
            ptr1 = int(self.bank_ptr1)
            if self.bank_full == True and self.bank_full1 == True:
                # feature_aug = self.attentive_interaction(self.bank, feats_flat, feats)
                feature_aug = self.attentive_interaction(self.bank, self.bank1, feats_flat, feats)
            else:
                # feature_aug = self.attentive_interaction(self.bank[0:ptr], feats_flat, feats)
                feature_aug = self.attentive_interaction(self.bank[0:ptr], self.bank1[0:ptr1], feats_flat, feats)

        elif flag == 'test':
            # feature_aug = self.attentive_interaction(patch, feats_flat, feats)
            feature_aug = self.attentive_interaction(patch, patch1, feats_flat, feats)
        # aux_out, patch, feats_flat, feats = self.region_representation(e5)
        # if flag == 'train':
        #     # print("flag", flag)
        #     self.update_bank(patch)
        #     ptr = int(self.bank_ptr)
        #     if self.bank_full == True:
        #         feature_aug = self.attentive_interaction(self.bank, feats_flat, feats)
        #     else:
        #         feature_aug = self.attentive_interaction(self.bank[0:ptr], feats_flat, feats)
        # elif flag == 'test':
        #     # print("flag", flag)
        #     feature_aug = self.attentive_interaction(patch, feats_flat, feats)

        # === Dual Attention === 内部注意力#
        # sa_feat1 = self.sa_head1(e5)
        # sa_feat2 = self.sa_head2(e5)
        # sa_feat = sa_feat1 + sa_feat2

        # === Fusion === 将外部注意和内部注意相加#
        # feats = sa_feat + feature_aug  # (4, 512, 8, 8)
        feats = feature_aug  # (4, 512, 8, 8)


        output = []
        for i in range(len(self.GCoutmodel)):  # 0 1 2 3
            output.append(self.GCoutmodel[i](feats))

        # Decoder5
        d5 = self.decoder5(feats)  # 14
        out5 = self.sideout5(d5)
        lc4 = self.lca4(e4, out5, e5)
        # lc4  = self.lca4(e4)
        # gc4 = global_contexts[0]
        gc4 = output[0]
        comb4 = self.asm4(lc4, d5, gc4) #16,16
        # comb4 = self.asm4(lc4, d5)

        # Decoder4
        d4 = self.decoder4(comb4)  # 28
        out4 = self.sideout4(d4)
        lc3 = self.lca3(e3, out4, lc4)
        # lc3 = self.lca3(e3, lc4)
        # lc3 = self.lca3(e3, e4)
        # gc3 = global_contexts[1]
        gc3 = output[1]
        comb3 = self.asm3(lc3, d4, gc3) #32,32
        # comb3 = self.asm3(lc3, d4)


        # Decoder3
        d3 = self.decoder3(comb3)  # 56
        out3 = self.sideout3(d3)
        lc2 = self.lca2(e2, out3, lc3)
        # lc2 = self.lca2(e2, lc3)
        # lc2 = self.lca2(e2, e3)
        # gc2 = global_contexts[2]
        gc2 = output[2]
        comb2 = self.asm2(lc2, d3, gc2) #64,64
        # comb2 = self.asm2(lc2, d3)


        # Decoder2
        d2 = self.decoder2(comb2)  # 128
        out2 = self.sideout2(d2)
        lc1 = self.lca1(e1, out2, lc2)
        # lc1 = self.lca1(e1, lc2)
        # lc1 = self.lca1(e1, e2)
        # gc1 = global_contexts[3]
        gc1 = output[3]
        comb1 = self.asm1(lc1, d2, gc1)
        # comb1 = self.asm1(lc1, d2)


        # Decoder1
        d1 = self.decoder1(comb1)  # 224*224*64
        out1 = self.outconv(d1)  # 224


        # APF
        # out_final = self.APF(d1, d2, d3)
        # out_final = self.outconv_final(out_final)

        #binary_attention
        # binary = self.binary_attention(comb4, comb3, comb2, e1)

        # return torch.sigmoid(out_final), torch.sigmoid(out1), torch.sigmoid(out2), torch.sigmoid(out3), \
        #     torch.sigmoid(out4), torch.sigmoid(out5)

        return torch.sigmoid(out1), torch.sigmoid(out2), torch.sigmoid(out3), \
               torch.sigmoid(out4), torch.sigmoid(out5) #, torch.sigmoid(binary)
