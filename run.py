import warnings
import os
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
from torchvision.models.segmentation import deeplabv3_resnet50
import torch
import torch.functional as F
import numpy as np
import requests
import torchvision
from PIL import Image
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam.grad_cam import GradCAM
import models
from opt import opt


# 读入自己的图像
image = np.array(Image.open('C:\\Users\\15059\\Desktop\\best_model\\Kvasir-SEG-image-result\\image\\0.jpg'))
mask = np.array(Image.open('C:\\Users\\15059\\Desktop\\best_model\\Kvasir-SEG-image-result\\mask\\0.jpg'))

# image = np.array(Image.open(image))
# mask = np.array(Image.open(mask))
rgb_img = np.float32(image) / 255
rgb_mask = np.float32(mask) / 255

tensor_img = preprocess_image(rgb_img,
                              mean=[0.5, 0.5, 0.5],
                              std=[0.129, 0.124, 0.125])
tensor_mask = preprocess_image(rgb_mask,
                                mean=[0.5],
                                std=[0.5])

input_tensor = torch.cat((tensor_img, tensor_mask), dim=0).unsqueeze(0)

# 读入自己的模型并且加载训练好的权重
# model = BiSeNet(backbone='STDCNet813', n_classes=6)
# model.cuda()
# model = model.eval()
# save_pth = '/media/wlj/soft_D/WLJ/WJJ/STDC-Seg/checkpoints/camera_4_crop/batch8_11.2_15000it_dublebaseline_left1xSGE2345_right0.5x_DFConv2_SGE3_RGB/model_maxmIOU100.pth'
# model.load_state_dict(torch.load(save_pth))

model = getattr(models, opt.model)(opt.nclasses)
model.cuda()
model = model.eval()
# load_ckpt_path = os.path.join('C:\\Users\\15059\\Desktop\\best_model\\Kvasir-SEG\\9.16 ACSNet 0.9146 0.9112\\ck_86.pth')
load_ckpt_path = os.path.join('C:\\Users\\15059\\Desktop\\model_data\\FACENet\\Kvasir-SEG\\ck_149.pth')
model.load_state_dict(torch.load(load_ckpt_path))

if torch.cuda.is_available():
    model = model.cuda()
    input_tensor = input_tensor.cuda()

# 推理
output = model(input_tensor, flag='test')[0]
normalized_masks = torch.softmax(output, dim=1).cpu()

# 自己的数据集的类别
# sem_classes = [
#     '__background__', 'round', 'nok', 'headbroken', 'headdeep', 'shoulderbroken'
# ]
sem_classes = [
    'ployp', '__background__',
]

sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(sem_classes)}
round_category = sem_class_to_idx["ployp"]
# round_mask = torch.argmax(normalized_masks[0], dim=0).detach().cpu().numpy()
# round_mask_uint8 = 255 * np.uint8(round_mask == round_category)
# round_mask_float = np.float32(round_mask == round_category)
plaque_mask = normalized_masks[0, :, :, :].argmax(axis=0).detach().cpu().numpy()
plaque_mask_uint8 = 255 * np.uint8(plaque_mask == round_category)
plaque_mask_float = np.float32(plaque_mask == round_category)

both_images = np.hstack((image, np.repeat(plaque_mask_uint8[:, :, None], 3, axis=-1)))
Image.fromarray(both_images)



# 推理结果图与原图拼接
# both_images = np.hstack((image, np.repeat(round_mask_uint8[:, :, None], 3, axis=-1)))
# img = Image.fromarray(both_images)
# img.save("./hhhh.png")


class SemanticSegmentationTarget:
    def __init__(self, category, mask):
        self.category = category
        self.mask = torch.from_numpy(mask)
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()

    def __call__(self, model_output):
        return (model_output[self.category, :, :] * self.mask).sum()


# 自己要放CAM的位置
target_layers = [model.outconv]
targets = [SemanticSegmentationTarget(round_category, plaque_mask_float)]

with GradCAM(model=model, target_layers=target_layers,
             use_cuda=torch.cuda.is_available()) as cam:
    grayscale_cam = cam(input_tensor=input_tensor,
                        targets=targets)[0, :]
    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

# 保存CAM的结果
img = Image.fromarray(cam_image)
img.show()
img.save('./result.png')