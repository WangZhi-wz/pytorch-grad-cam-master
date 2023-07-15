import io
import warnings
import os
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
from torchvision.models.segmentation import deeplabv3_resnet50
import torch
import torchvision.transforms.functional as F
import numpy as np
import requests
import torchvision
from PIL import Image
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam.grad_cam import GradCAM
import models
from opt import opt
import skimage.io as io


# 读入自己的图像
image = Image.open('C:\\Users\\15059\\Desktop\\best_model\\Kvasir-SEG-image-result\\image\\0.jpg').convert('RGB')  # 320,320,3
# io.imsave(os.path.join('./result/image.jpg'), image)


rgb_img = np.float32(image) / 255 # 320,320,3
input_tensor = preprocess_image(rgb_img,
                                mean=[0.5, 0.5, 0.5],
                                std=[0.5, 0.5, 0.5])


# 读入自己的模型并且加载训练好的权重
model = getattr(models, opt.model)(opt.nclasses)
model.cuda()
model = model.eval()
model_dict = model.state_dict()
# load_ckpt_path = os.path.join('C:\\Users\\15059\\Desktop\\best_model\\Kvasir-SEG\\9.16 ACSNet 0.9146 0.9112\\ck_86.pth')
load_ckpt_path = os.path.join('C:\\Users\\15059\\Desktop\\model_data\\FACENet\\Kvasir-SEG\\ck_153.pth')
# load_ckpt_path = os.path.join('C:\\Users\\15059\\Desktop\\best_model\\Kvasir-SEG\\9.19 UACANet 0.9136 0.9124\\ck_149.pth')
# load_ckpt_path = os.path.join(r'C:\Users\15059\Desktop\best_model\Kvasir-SEG-val--test-best\10.1 CCBANet\ck_165.pth')
# load_ckpt_path = os.path.join(r'C:\Users\15059\Desktop\best_model\Kvasir-SEG-val--test-best\10.19 UNet\ck_200.pth')
print(load_ckpt_path)
assert os.path.isfile(load_ckpt_path), 'No checkpoint found.'
print('Loading checkpoint......')
checkpoint = torch.load(load_ckpt_path)
new_dict = {k: v for k, v in checkpoint.items() if k in model_dict.keys()}
model_dict.update(new_dict)
model.load_state_dict(model_dict)
if torch.cuda.is_available():
    model = model.cuda()
    input_tensor = input_tensor.cuda()



# 推理
if opt.model == 'ACSNet_caRABAsaBD_modDCR':
    output = model(input_tensor, flag='test')[0]
else:
    output = model(input_tensor)[0]

print(output)
predict = torch.squeeze(output[0]).detach().cpu().numpy()    # 320,320
predict[predict > 0.5] = 255
predict[predict <= 0.5] = 0
predict = np.expand_dims(predict, axis=2)
predict = np.squeeze(predict)
predict = [predict, predict, predict]
predict = np.transpose(predict, (1, 2, 0))
io.imsave(os.path.join('./result/predict.jpg'), predict)

normalized_masks = torch.softmax(output, dim=1).cpu()

# 自己的数据集的类别
# sem_classes = [
#     '__background__', 'round', 'nok', 'headbroken', 'headdeep', 'shoulderbroken'
# ]
sem_classes = [
     '__background__', 'ployp'
]


sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(sem_classes)}
round_category = sem_class_to_idx['__background__']
round_mask = torch.argmax(normalized_masks[0], dim=0).detach().cpu().numpy()    # [320,320]
round_mask_uint8 = 255 * np.uint8(round_mask == round_category)
round_mask_float = np.float32(round_mask == round_category)


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

targets = [SemanticSegmentationTarget(round_category, round_mask_float)]

with GradCAM(model=model, target_layers=target_layers,
             use_cuda=torch.cuda.is_available()) as cam:
    grayscale_cam = cam(input_tensor=input_tensor,
                        targets=targets)[0, :]
    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)



# 保存CAM的结果
img = Image.fromarray(cam_image)
img.show()
img.save('./result/featuremap.png')