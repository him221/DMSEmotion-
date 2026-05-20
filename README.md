# DMSEmotion - 基于多模态的驾乘人员情绪监测系统

> Open-source note: large model files and datasets are intentionally not included in this repository. Before running the GUI, read `OPEN_SOURCE_MODEL_FILES.md` and place the required model files in the documented paths.

## Model Download

This repository contains source code, documentation, small configuration files, tokenizers, and UI assets only. The runtime model weights are distributed separately through GitHub Releases.

Primary download:

- GitHub Releases: `https://github.com/<your-github-username>/DMSEmotion/releases/tag/v1.0-models`

Domestic mirror, optional:

- Baidu Netdisk: `TODO: add your Baidu Netdisk link`
- Extraction code: `TODO`

Download these three files from the release page:

| File | Destination |
| --- | --- |
| `PrivateTest_model.t7` | `Expression Recognition/FER2013_VGG19/PrivateTest_model.t7` |
| `shape_predictor_68_face_landmarks.dat` | `Expression Recognition/shape_predictor_68_face_landmarks.dat` |
| `model.safetensors` | `Chinese classification/results/checkpoint-86050/model.safetensors` |

After placing the files, install dependencies and run:

```powershell
pip install -r requirements.txt
cd "Expression Recognition"
python app_gui2.py
```

Speech recognition is optional. To enable it, set your Xunfei credentials before starting the GUI:

```powershell
$env:XFYUN_APPID="your_appid"
$env:XFYUN_API_KEY="your_api_key"
$env:XFYUN_API_SECRET="your_api_secret"
```

<div align="center">

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-orange.svg)
![Transformers](https://img.shields.io/badge/Transformers-4.0+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

一个集成了面部表情识别、语音情感分析和中文文本情感分类的多模态情绪监测系统

[功能特性](#功能特性) • [快速开始](#快速开始) • [项目结构](#项目结构) • [使用说明](#使用说明) • [模型说明](#模型说明)

</div>

---

## 📋 目录

- [项目简介](#项目简介)
- [功能特性](#功能特性)
- [技术栈](#技术栈)
- [快速开始](#快速开始)
- [项目结构](#项目结构)
- [使用说明](#使用说明)
- [模型说明](#模型说明)
- [数据集](#数据集)
- [实验结果](#实验结果)
- [贡献指南](#贡献指南)
- [许可证](#许可证)

---

## 🎯 项目简介

DMSEmotion 是一个面向驾乘人员的多模态情绪监测系统，通过融合**面部表情识别**、**语音情感分析**和**中文文本情感分类**三种模态，实现对驾乘人员情绪状态的实时监测与分析。

### 核心功能

- 🎭 **面部表情识别**：基于 VGG19 深度学习模型，识别 7 种基本情绪（快乐、愤怒、悲伤、恐惧、厌恶、惊讶、中性）
- 🎤 **实时语音转文字**：集成讯飞语音识别 API，实现实时语音转文字
- 📝 **中文文本情感分析**：基于 BERT/RoBERTa 微调模型，对中文文本进行情感分类
- 🔄 **多模态融合**：融合面部和语音文本情绪，提供更准确的情绪判断
- 📊 **可视化分析**：提供情绪轮、概率分布、效价-唤醒度分析、时间轴等多种可视化方式

---

## ✨ 功能特性

### 1. 表情识别模块

- **单张图片识别**：上传图片进行表情分析
- **实时视频识别**：通过摄像头实时捕捉并分析面部表情
- **FACS 编码分析**：基于面部动作编码系统（Facial Action Coding System）分析面部肌肉运动
- **情绪空间定位**：在效价-唤醒度（Valence-Arousal）二维空间中定位情绪

### 2. 文本情感分析模块

支持多种方法进行中文文本情感分类：

- **词典法**：基于 HowNet 词典的情感极性计算
- **BERT 微调**：使用 bert-base-chinese 预训练模型微调
- **RoBERTa 微调**：使用 XLM-RoBERTa 预训练模型微调
- **Prompt 方法**：基于提示学习的零样本情感分类

### 3. 多模态融合

- 融合面部表情和语音文本情绪
- 生成综合情绪报告
- 提供情绪分布统计和趋势分析

### 4. 数据记录与分析

- 实时记录情绪数据（CSV/JSON 格式）
- 生成情绪变化趋势图
- 导出分析报告

---

## 🛠 技术栈

### 深度学习框架
- PyTorch
- TensorFlow/Keras
- Transformers (Hugging Face)

### 计算机视觉
- OpenCV
- dlib
- PIL

### 自然语言处理
- BERT (bert-base-chinese)
- XLM-RoBERTa
- cnsenti (中文情感分析库)

### GUI 开发
- Tkinter
- Matplotlib

### 其他
- pandas, numpy
- scikit-learn
- websocket (讯飞语音 API)

---

## 🚀 快速开始

### 环境要求

- Python 3.7+
- CUDA 10.2+ (推荐，用于 GPU 加速)
- Windows/Linux/MacOS

### 安装步骤

1. **克隆仓库**

```bash
git clone https://github.com/yourusername/DMSEmotion.git
cd DMSEmotion
```

2. **安装依赖**

```bash
pip install -r requirements.txt
```

3. **下载预训练模型**

⚠️ **重要**: 由于 GitHub 文件大小限制，大型模型文件需要单独下载。

请查看 **[MODEL_DOWNLOAD.md](MODEL_DOWNLOAD.md)** 获取详细的模型下载指南。

**快速下载**：
- [dlib 人脸关键点模型](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) (95MB)
- BERT 模型：`pip install transformers` 后自动下载
- 其他模型：见 [MODEL_DOWNLOAD.md](MODEL_DOWNLOAD.md)

4. **配置讯飞语音 API**（可选）

在 `Expression Recognition/app_gui2.py` 中修改以下参数：

```python
self.speech_client = IatWsClient(
    appid='your_appid',
    api_key='your_api_key',
    api_secret='your_api_secret',
    text_callback=debug_cb
)
```

5. **运行程序**

```bash
cd "Expression Recognition"
python app_gui2.py
```

---

## 📁 项目结构

```
DMSEmotion/
├── Expression Recognition/          # 表情识别模块
│   ├── app_gui2.py                 # 主程序 GUI
│   ├── real_time_recognition.py    # 实时识别核心
│   ├── image_recognition.py        # 图片识别
│   ├── facs_analysis.py            # FACS 编码分析
│   ├── visualization.py            # 可视化工具
│   ├── multimodal_fusion.py        # 多模态融合
│   ├── text_emotion_analysis.py    # 文本情感分析
│   ├── iat_ws_client.py            # 讯飞语音 API 客户端
│   ├── dynamic_linechart.py        # 动态折线图
│   ├── dynamic_barchart.py         # 动态柱状图
│   ├── dynamic_textlog.py          # 时间轴日志
│   ├── Realtime_Data_Save.py       # 数据记录
│   ├── FER2013_VGG19/              # FER2013 训练模型
│   ├── CK+_VGG19/                  # CK+ 训练模型
│   └── shape_predictor_68_face_landmarks.dat  # dlib 人脸关键点模型
│
├── Chinese classification/          # 中文情感分类模块
│   ├── bert_fine_tune.py           # BERT 微调训练
│   ├── RoBERTa_fine_tune.py        # RoBERTa 微调训练
│   ├── predict.py                  # 预测脚本
│   ├── predict_by_bert.py          # BERT 预测
│   ├── predict_by_Robert.py        # RoBERTa 预测
│   ├── bert-base-chinese/          # BERT 预训练模型
│   ├── emotion_chinese_english/    # 情感分类模型
│   ├── data/                       # 训练数据集
│   ├── model/                      # 训练好的模型
│   ├── results/                    # 训练结果
│   ├── utility/                    # 工具函数
│   │   ├── bert_utility.py
│   │   └── RoBERTa_utility.py
│   └── ipynb/                      # Jupyter Notebook 实验
│
└── Data Recording/                  # 数据记录目录
    ├── csv-data/                   # CSV 格式数据
    ├── json-data/                  # JSON 格式数据
    └── png-data/                   # 图表数据
```

---

## 📖 使用说明

### 1. 单张图片识别

1. 点击"上传图片"按钮
2. 选择要分析的图片
3. 系统将显示：
   - 识别结果和置信度
   - FACS 编码分析
   - 情绪空间位置（效价-唤醒度）
   - 情绪轮分析
   - 表情概率分布
   - Emoji 形象

### 2. 实时识别

1. 点击"实时识别"按钮
2. 系统将启动摄像头和语音识别
3. 实时显示：
   - 面部表情识别结果
   - 5 分钟内情绪变化趋势
   - 情绪占比分布
   - 情绪时间轴滚动日志
   - 实时语音转文字
4. 点击"保存数据"保存识别结果
5. 点击"数据分析"进行文本情感分析和多模态融合

### 3. 结束识别

点击"结束识别"按钮停止当前识别任务

---

## 🧠 模型说明

### 表情识别模型

- **架构**：VGG19
- **数据集**：FER2013 / CK+
- **类别**：7 种情绪（Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral）
- **输入**：48x48 灰度图像
- **准确率**：约 65-70%

### 中文文本情感分类模型

#### BERT 模型
- **预训练模型**：bert-base-chinese
- **微调数据集**：天池情绪分类数据集
- **类别**：5 种情绪（sadness, happiness, anger, fear, disgust）
- **最大序列长度**：128
- **F1 Score**：约 0.64-0.65

#### RoBERTa 模型
- **预训练模型**：XLM-RoBERTa
- **支持**：中英文双语
- **训练参数**：
  - Batch size: 64
  - Learning rate: 2e-5
  - Epochs: 15

---

## 📊 数据集

### 表情识别数据集

1. **FER2013**
   - 包含约 35,000 张 48x48 灰度人脸图像
   - 7 种情绪类别
   - 公开数据集

2. **CK+ (Extended Cohn-Kanade)**
   - 高质量实验室环境采集
   - 包含 FACS 编码标注
   - 适合精细化表情分析

### 文本情感数据集

1. **天池情绪分类数据集**
   - 中文微博情绪数据
   - 5 种情绪类别
   - 约 27,000+ 训练样本

2. **OCEMOTION 数据集**
   - 中文情感语料
   - 多种情绪标注

3. **大连理工情感词典**
   - 用于词典法情感分析
   - 包含情感极性值

---

## 🔬 实验结果

### 表情识别性能

| 模型 | 数据集 | 准确率 | F1 Score |
|------|--------|--------|----------|
| VGG19 | FER2013 | ~68% | ~0.66 |
| VGG19 | CK+ | ~92% | ~0.91 |

### 文本情感分类性能

| 模型 | F1 Score (Weighted) |
|------|---------------------|
| BERT (微调) | 0.6414 - 0.6521 |
| RoBERTa (微调) | ~0.65 |
| 词典法 | ~0.55 |

### 多模态融合

融合面部表情和语音文本情绪后，系统能够提供更全面的情绪判断，特别是在以下场景：
- 面部表情不明显但语音情绪强烈
- 语音平静但面部表情激动
- 两种模态情绪一致时提高置信度

---

## 🎨 可视化示例

系统提供多种可视化方式：

1. **情绪轮分析**：雷达图展示 7 种情绪的强度
2. **表情概率分布**：柱状图显示各情绪的概率
3. **效价-唤醒度分析**：二维空间定位情绪
4. **情绪变化趋势**：折线图展示 5 分钟内情绪变化
5. **情绪时间轴**：滚动日志记录每次情绪识别
6. **Emoji 形象**：直观展示当前情绪

---

## 🔧 自定义训练

### 训练表情识别模型

```bash
cd "Expression Recognition"
# FER2013 数据集
python mainpro_FER.py

# CK+ 数据集
python mainpro_CK+.py
```

### 训练文本情感分类模型

```bash
cd "Chinese classification"
# BERT 微调
python bert_fine_tune.py

# RoBERTa 微调
python RoBERTa_fine_tune.py
```

---

## 🤝 贡献指南

欢迎贡献代码、报告问题或提出新功能建议！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

---

## 📝 待办事项

- [ ] 添加更多情绪类别支持
- [ ] 优化多模态融合算法
- [ ] 支持更多语言的文本情感分析
- [ ] 添加情绪预警功能
- [ ] 优化实时性能
- [ ] 添加云端部署方案
- [ ] 完善文档和使用教程

---

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

---

## 🙏 致谢

- [FER2013 数据集](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge)
- [CK+ 数据集](http://www.consortium.ri.cmu.edu/ckagree/)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [dlib](http://dlib.net/)
- [讯飞开放平台](https://www.xfyun.cn/)

---

## 📧 联系方式

如有问题或建议，请通过以下方式联系：

- 提交 Issue
- 发送邮件至：[your-email@example.com]

---

<div align="center">

**如果这个项目对你有帮助，请给个 ⭐️ Star 支持一下！**

Made with ❤️ by [Your Name]

</div>
