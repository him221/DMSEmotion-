import torch
import torch.nn.functional as F
import transforms as transforms
import numpy as np
import os
from PIL import Image
from skimage import io
from skimage.transform import resize
from models import VGG  # 这里确保导入你的VGG模型定义

# 与原代码一致
class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
ncrops = 10  # 因为使用了TenCrop

def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

def process_image(image_path):
    # 原始的图像读取与处理逻辑
    raw_img = io.imread(image_path)
    gray = rgb2gray(raw_img)
    gray = resize(gray, (48, 48), mode='symmetric').astype(np.uint8)
    img = np.stack((gray,) * 3, axis=-1)
    img = Image.fromarray(img)
    transform_test = transforms.Compose([
        transforms.TenCrop(44),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]))
    ])
    inputs = transform_test(img)
    return inputs, raw_img

def predict_expression(image_path):
    """
    1. 调用 process_image
    2. 加载 VGG19 模型
    3. 预测表情并返回 (predicted_class, scores, raw_img)
    """
    inputs, raw_img = process_image(image_path)

    # 加载模型
    # 确保路径正确，如 FER2013_VGG19/PrivateTest_model.t7
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = VGG('VGG19')
    checkpoint = torch.load(os.path.join('FER2013_VGG19', 'PrivateTest_model.t7'), map_location=device)
    net.load_state_dict(checkpoint['net'])
    net.to(device)
    net.eval()

    # 执行推理
    inputs = inputs.view(-1, *inputs.shape[-3:]).to(device)
    with torch.no_grad():
        outputs = net(inputs)

    outputs_avg = outputs.view(ncrops, -1).mean(0)
    scores = F.softmax(outputs_avg, dim=0)
    predicted = torch.argmax(outputs_avg).item()

    return class_names[predicted], scores.cpu().numpy(), raw_img
