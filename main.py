import os
import time

import numpy as np
from torchvision.transforms import Resize

os.environ['CUDA_ENABLE_DEVICES'] = '0'
import torch
import torch.nn
import torch.optim as optim
from torchvision import transforms, models
import CLIPTrans.TaylorFormer as MBformer
import utils
import clip
import torch.nn.functional as F
from template import imagenet_templates

from torchvision import utils as vutils
import argparse
from torchvision.transforms.functional import adjust_contrast


parser = argparse.ArgumentParser()

parser.add_argument('--content_path', type=str, default="./face.jpg",
                    help='Image resolution')
parser.add_argument('--content_name', type=str, default="face",
                    help='Image resolution')
parser.add_argument('--exp_name', type=str, default="exp1",
                    help='Image resolution')
parser.add_argument('--text', type=str, default="Fire",
                    help='Image resolution')
parser.add_argument('--lambda_tv', type=float, default=2e-3,
                    help='total variation loss parameter')
parser.add_argument('--lambda_patch', type=float, default=9000,
                    help='PatchCLIP loss parameter')
parser.add_argument('--lambda_dir', type=float, default=500,
                    help='directional loss parameter')
parser.add_argument('--lambda_c', type=float, default=150,
                    help='content loss parameter')
parser.add_argument('--crop_size', type=int, default=128,
                    help='cropped image size')
parser.add_argument('--num_crops', type=int, default=64,
                    help='number of patches')
parser.add_argument('--img_width', type=int, default=512,
                    help='size of images')
parser.add_argument('--img_height', type=int, default=512,
                    help='size of images')
parser.add_argument('--max_step', type=int, default=200,
                    help='Number of domains')
parser.add_argument('--lr', type=float, default=5e-4,
                    help='Number of domains')
parser.add_argument('--thresh', type=float, default=0.7,
                    help='Number of domains')

parser.add_argument('--nfc', type=int, default=16)
parser.add_argument('--nc_im', type=int, help='image # channels', default=3)
parser.add_argument('--ker_size', type=int, help='kernel size', default=3)
parser.add_argument('--padd_size', type=int, help='net pad size', default=1)
parser.add_argument('--min_nfc', type=int, default=3)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

assert (args.img_width%8)==0, "width must be multiple of 8"
assert (args.img_height%8)==0, "height must be multiple of 8"

# VGG Features
VGG = models.vgg19(pretrained=True).features
VGG.to(device)

# VGG权重不参与训练
for parameter in VGG.parameters():
    parameter.requires_grad_(False)


def img_denormalize(image):
    mean=torch.tensor([0.485, 0.456, 0.406]).to(device)
    std=torch.tensor([0.229, 0.224, 0.225]).to(device)
    mean = mean.view(1,-1,1,1)
    std = std.view(1,-1,1,1)

    image = image*std +mean
    return image


def img_normalize(image):
    mean=torch.tensor([0.485, 0.456, 0.406]).to(device)
    std=torch.tensor([0.229, 0.224, 0.225]).to(device)
    mean = mean.view(1,-1,1,1)
    std = std.view(1,-1,1,1)

    image = (image-mean)/std
    return image


def clip_normalize(image,device):
    image = F.interpolate(image,size=224,mode='bicubic')
    mean=torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(device)
    std=torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(device)
    mean = mean.view(1,-1,1,1)
    std = std.view(1,-1,1,1)

    image = (image-mean)/std
    return image

    
def get_image_prior_losses(inputs_jit):
    diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
    diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
    diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
    diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]

    loss_var_l2 = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
    
    return loss_var_l2

def compose_text_with_templates(text: str, templates=imagenet_templates) -> list:
    return [template.format(text) for template in templates]

# 加载内容图像和内容名字
content_path = args.content_path
content_image = utils.load_image2(content_path,img_height=args.img_height,img_width=args.img_width)

content = args.content_name
exp = args.exp_name

content_image = content_image.to(device)

# 提取图像内容特征
content_features = utils.get_features(img_normalize(content_image), VGG)
target = content_image.clone().requires_grad_(True).to(device)

# 全局特征
transformer_net = MBformer.Generator(args).to(device)


style_weights = {'conv1_1': 0.1,
                 'conv2_1': 0.2,
                 'conv3_1': 0.4,
                 'conv4_1': 0.8,
                 'conv5_1': 1.6}

content_weight = args.lambda_c

show_every = 100
optimizer = optim.Adam(transformer_net.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
steps = args.max_step

content_loss_epoch = []
style_loss_epoch = []
total_loss_epoch = []

output_image = content_image

m_cont = torch.mean(content_image,dim=(2,3),keepdim=False).squeeze(0)
m_cont = [m_cont[0].item(),m_cont[1].item(),m_cont[2].item()]

cropper = transforms.Compose([
    transforms.RandomCrop(args.crop_size)
])
augment = transforms.Compose([
    transforms.RandomPerspective(fill=0, p=1,distortion_scale=0.5),
    transforms.Resize(224)
])
device='cuda'

# 记载clip模型
clip_model, preprocess = clip.load('ViT-B/32', device, jit=False)
# 内容信息
prompt = args.text

source = "a Photo"

begin = time.time()

with torch.no_grad():
    # 目标风格文本信息CLIP特征提取
    template_text = compose_text_with_templates(prompt, imagenet_templates)
    tokens = clip.tokenize(template_text).to(device)
    text_features = clip_model.encode_text(tokens).detach()
    text_features = text_features.mean(axis=0, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # 图像内容文本信息CLIP特征提取
    template_source = compose_text_with_templates(source, imagenet_templates)
    tokens_source = clip.tokenize(template_source).to(device)
    text_source = clip_model.encode_text(tokens_source).detach()
    text_source = text_source.mean(axis=0, keepdim=True)
    text_source /= text_source.norm(dim=-1, keepdim=True)
    # 图像内容CLIP图像特征提取
    source_features = clip_model.encode_image(clip_normalize(content_image,device))
    source_features /= (source_features.clone().norm(dim=-1, keepdim=True))

# number of patches
num_crops = args.num_crops
for epoch in range(0, steps+1):
    # 风格迁移过程
    scheduler.step()
    target = transformer_net(content_image).to(device)
    target.requires_grad_(True)

    # 风格化结果特征提取
    target_features = utils.get_features(img_normalize(target), VGG)

    # 内容损失
    content_loss = 0

    content_loss += torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)
    content_loss += torch.mean((target_features['conv5_2'] - content_features['conv5_2']) ** 2)

    # PatchLoss损失
    loss_patch=0
    img_proc =[]
    for n in range(num_crops):
        target_crop = cropper(target)
        target_crop = augment(target_crop)
        img_proc.append(target_crop)

    img_proc = torch.cat(img_proc,dim=0)
    img_aug = img_proc
    # 风格图像剪切增强后级联后CLIP图像特征提取
    image_features = clip_model.encode_image(clip_normalize(img_aug,device))
    image_features /= (image_features.clone().norm(dim=-1, keepdim=True))
    #
    img_direction = (image_features-source_features)
    img_direction /= img_direction.clone().norm(dim=-1, keepdim=True)

    text_direction = (text_features-text_source).repeat(image_features.size(0),1)
    text_direction /= text_direction.norm(dim=-1, keepdim=True)
    loss_temp = (1 - torch.cosine_similarity(img_direction, text_direction, dim=1))
    loss_temp[loss_temp<args.thresh] =0
    loss_patch+=loss_temp.mean()

    #  Ldir 损失
    glob_features = clip_model.encode_image(clip_normalize(target,device))
    glob_features /= (glob_features.clone().norm(dim=-1, keepdim=True))
    
    glob_direction = (glob_features-source_features)
    glob_direction /= glob_direction.clone().norm(dim=-1, keepdim=True)
    
    loss_glob = (1 - torch.cosine_similarity(glob_direction, text_direction, dim=1)).mean()

    # Ltv损失
    reg_tv = args.lambda_tv*get_image_prior_losses(target)

    # 自己新添加的身份损失，保持源数据不变
    loss = torch.nn.L1Loss()
    # Z_opt2 = m_image(source)
    # print(type(source))
    # print(source.shape)
    # print(target.shape)
    rec_loss = loss(content_image,target)

    # total_loss = args.lambda_patch*loss_patch + content_weight * content_loss + reg_tv+ args.lambda_dir*loss_glob
    total_loss = args.lambda_patch*loss_patch + content_weight * content_loss+ reg_tv+ args.lambda_dir*loss_glob + 0.1 * rec_loss
    # total_loss = args.lambda_patch*loss_patch + reg_tv+ args.lambda_dir*loss_glob + 0.1 * rec_loss
    total_loss_epoch.append(total_loss)

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print("After %d criterions:" % epoch)
        print('Total loss: ', total_loss.item())
        print('Content loss: ', content_loss.item())
        print('ID loss: ', rec_loss.item())
        print('patch loss: ', loss_patch.item())
        print('dir loss: ', loss_glob.item())
        print('TV loss: ', reg_tv.item())
        print('rec loss: ', rec_loss.item())

    if epoch %50 ==0:
        out_path = './long/'+prompt+'_'+content+'_'+exp+'.png'
        output_image = target.clone()
        output_image = torch.clamp(output_image,0,1)
        output_image = adjust_contrast(output_image,1.5)
        vutils.save_image(
                                    output_image,
                                    out_path,
                                    nrow=1,
                                    normalize=True)



end = time.time()
print("the total of time is: ", (end-begin), " s")
