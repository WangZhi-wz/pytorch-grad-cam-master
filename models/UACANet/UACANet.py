import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# from .optim.losses import *

# from .modules.context_module import *
# from .modules.attention_module import *


from models.UACANet.modules.attention_module import UACA
from models.UACANet.modules.context_module import PAA_e
from models.UACANet.modules.decoder_module import *

from models.UACANet.backbones.Res2Net_v1b import res2net50_v1b_26w_4s
from models.UACANet.optim import bce_iou_loss


class UACANet(nn.Module):
    # res2net based encoder decoder
    def __init__(self, channels=256, output_stride=16, pretrained=True):
        super(UACANet, self).__init__()
        self.resnet = res2net50_v1b_26w_4s(pretrained=pretrained, output_stride=output_stride)
        channels = 256

        self.context2 = PAA_e(512, channels)
        self.context3 = PAA_e(1024, channels)
        self.context4 = PAA_e(2048, channels)

        self.paa_d = PAA_d(channels)

        self.attention2 = UACA(channels * 2, channels)
        self.attention3 = UACA(channels * 2, channels)
        self.attention4 = UACA(channels * 2, channels)

        self.loss_fn = bce_iou_loss

        self.ret = lambda x, target: F.interpolate(x, size=target.shape[-2:], mode='bilinear', align_corners=False)
        self.res = lambda x, size: F.interpolate(x, size=size, mode='bilinear', align_corners=False)

    def forward(self, sample):
        x = sample
        # y = gt
        # x = sample['image']
        # if 'gt' in sample.keys():
        #     y = sample['gt']
        # else:
        #     y = None
            
        base_size = x.shape[-2:]
        
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x1 = self.resnet.layer1(x)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)

        x2 = self.context2(x2)
        x3 = self.context3(x3)
        x4 = self.context4(x4)

        f5, a5 = self.paa_d(x4, x3, x2)
        out5 = self.res(a5, base_size)

        f4, a4 = self.attention4(torch.cat([x4, self.ret(f5, x4)], dim=1), a5)
        out4 = self.res(a4, base_size)

        f3, a3 = self.attention3(torch.cat([x3, self.ret(f4, x3)], dim=1), a4)
        out3 = self.res(a3, base_size)

        _, a2 = self.attention2(torch.cat([x2, self.ret(f3, x2)], dim=1), a3)
        out2 = self.res(a2, base_size)

        return torch.sigmoid(out5), torch.sigmoid(out4), torch.sigmoid(out3), torch.sigmoid(out2)


        # if y is not None:
        #     loss5 = self.loss_fn(out5, y)
        #     loss4 = self.loss_fn(out4, y)
        #     loss3 = self.loss_fn(out3, y)
        #     loss2 = self.loss_fn(out2, y)
        #
        #     loss = loss2 + loss3 + loss4 + loss5
        #     debug = [out5, out4, out3]
        # else:
        #     loss = 0
        #     debug = []
        # return {'pred': out2, 'loss': loss, 'debug': debug}

if __name__ == '__main__':
    input=torch.randn(4,3,256,256)
    gt= torch.randn(4,1,256,256)
    # input = torch.randn(50, 512, 7, 7)
    # input = "D:\\datasets\\Kvasir1\\train\\images\\cju0qkwl35piu0993l0dewei2.jpg"

    ua = UACANet()
    output = ua(input)



    # print(output.shape)