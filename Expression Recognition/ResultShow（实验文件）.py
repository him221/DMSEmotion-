import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import Label, Button, Frame
from PIL import Image
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import *
import transforms as transforms
from torch.autograd import Variable
import os
from skimage import io
from skimage.transform import resize

# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 使用微软雅黑字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

class_names = ['愤怒', '厌恶', '恐惧', '快乐', '悲伤', '惊讶', '中性']

# 定义图像处理函数
def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def process_image(image_path):
    raw_img = io.imread(image_path)
    gray = rgb2gray(raw_img)
    gray = resize(gray, (48, 48), mode='symmetric').astype(np.uint8)
    img = gray[:, :, np.newaxis]
    img = np.concatenate((img, img, img), axis=2)
    img = Image.fromarray(img)
    transform_test = transforms.Compose([
        transforms.TenCrop(44),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]))
    ])
    inputs = transform_test(img)
    return inputs, raw_img


# 定义模型预测函数
def predict_expression(image_path, model):
    inputs, raw_img = process_image(image_path)

    class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    net = VGG('VGG19')
    checkpoint = torch.load(os.path.join('FER2013_VGG19', 'PrivateTest_model.t7'))
    net.load_state_dict(checkpoint['net'])
    net.cuda()
    net.eval()

    ncrops, c, h, w = np.shape(inputs)
    inputs = inputs.view(-1, c, h, w)
    inputs = inputs.cuda()

    # 修改这部分，使用 with torch.no_grad():
    with torch.no_grad():  # 禁用梯度计算
        outputs = net(inputs)

    outputs_avg = outputs.view(ncrops, -1).mean(0)  # avg over crops
    score = F.softmax(outputs_avg, dim=0)
    _, predicted = torch.max(outputs_avg.data, 0)

    return class_names[int(predicted.cpu().numpy())], score



# 创建窗口
class EmotionRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title('表情识别系统')
        self.root.geometry('1920x1080')
        # self.root.attributes("-fullscreen", True)
        self.root.configure(bg='#f0f0f0')  # 设置背景颜色

        # 添加标题
        self.title_label = Label(root, text="人脸表情识别系统", font=("Microsoft YaHei", 24, 'bold'), bg='#f0f0f0', fg="#333")
        self.title_label.pack(pady=20)

        # 介绍文字
        self.info_label = Label(root, text="请选择一张图片来进行表情分类，系统将预测表情并显示分类结果。",
                                font=("Microsoft YaHei", 14), bg='#f0f0f0', fg="#555")
        self.info_label.pack(pady=10)

        # 上传按钮
        self.upload_button = Button(root, text="上传图片", font=("Microsoft YaHei", 14), command=self.upload_image,
                                    relief="raised", width=20, height=2, bg='#5cb85c', fg='white')
        self.upload_button.pack(pady=20)

        # 结果标签
        self.result_label = Label(root, text="识别结果：", font=("Microsoft YaHei", 14), bg='#f0f0f0', fg="#333")
        self.result_label.pack(pady=10)

        # 创建一个画布用于显示图片
        self.canvas = None  # 用来显示图像
        self.result_text = Label(root, text="未选择图片", font=("Microsoft YaHei", 14), bg='#f0f0f0', fg="#333")
        self.result_text.pack(pady=20)  # 用于显示预测结果

        # 监听关闭窗口事件，确保程序退出
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def on_close(self):
        print("程序关闭")
        self.root.quit()  # 退出事件循环
        self.root.destroy()  # 销毁窗口

    def upload_image(self):
        file_path = filedialog.askopenfilename(title="选择图片", filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
        if file_path:
            self.show_result(file_path)

    def show_result(self, file_path):
        # 清除画布之前显示的内容
        if self.canvas:
            self.canvas.get_tk_widget().destroy()  # 销毁旧的画布

        # 获取预测结果
        predicted_class, score = predict_expression(file_path, None)

        # 结果显示
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        # 显示输入图片
        raw_img = io.imread(file_path)
        axes[0].imshow(raw_img)
        axes[0].set_title("输入图片", fontsize=16, fontweight='bold')
        axes[0].axis('off')

        # 显示分类结果
        ind = 0.1 + 0.6 * np.arange(len(score))
        width = 0.4
        color_list = ['red', 'orangered', 'darkorange', 'limegreen', 'darkgreen', 'royalblue', 'navy']
        for i in range(len(score)):
            axes[1].bar(ind[i], score.data.cpu().numpy()[i], width, color=color_list[i])
        axes[1].set_title("分类结果", fontsize=16, fontweight='bold')
        axes[1].set_xlabel("表情种类", fontsize=14)
        axes[1].set_ylabel("分类得分", fontsize=14)
        axes[1].set_xticks(ind)
        axes[1].set_xticklabels(['愤怒', '厌恶', '恐惧', '快乐', '悲伤', '惊讶', '中性'], rotation=45)

        # 显示表情符号
        emojis_img = io.imread(f'images/emojis/{predicted_class}.png')
        axes[2].imshow(emojis_img)
        axes[2].set_title("表情符号", fontsize=16, fontweight='bold')
        axes[2].axis('off')

        plt.tight_layout()
        plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1)

        # 将matplotlib图像嵌入Tkinter窗口
        self.canvas = FigureCanvasTkAgg(fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(pady=20)

        # 更新识别结果文本
        self.result_text.config(
            text=f"识别结果: {predicted_class} ({score.data.cpu().numpy()[class_names.index(predicted_class)] * 100:.2f}%)")


# 创建Tkinter窗口
root = tk.Tk()
app = EmotionRecognitionApp(root)
root.mainloop()