# 模型下载指南

由于 GitHub 文件大小限制，大型模型文件需要单独下载。请按照以下步骤配置项目所需的模型。

---

## 📥 必需模型（运行项目必须下载）

### 1. BERT 中文预训练模型

**模型名称**: bert-base-chinese  
**大小**: 约 400 MB  
**用途**: 中文文本情感分类

#### 下载方式 A：从 Hugging Face 下载（推荐）

```bash
# 使用 Python 自动下载
python -c "from transformers import BertTokenizer, BertForSequenceClassification; BertTokenizer.from_pretrained('bert-base-chinese'); BertForSequenceClassification.from_pretrained('bert-base-chinese')"
```

下载后的文件会自动保存到缓存目录。

#### 下载方式 B：手动下载

1. 访问：https://huggingface.co/bert-base-chinese/tree/main
2. 下载以下文件到 `Chinese classification/bert-base-chinese/` 目录：
   - `pytorch_model.bin` (392 MB) ⭐ 必需
   - `config.json` ✅ 已包含
   - `vocab.txt` ✅ 已包含
   - `tokenizer.json` ✅ 已包含
   - `tokenizer_config.json` ✅ 已包含

#### 下载方式 C：百度网盘

- **下载链接**: [待上传]
- **提取码**: xxxx
- **解压到**: `Chinese classification/bert-base-chinese/`

---

### 2. 情感分类模型（中英文）

**模型名称**: emotion_chinese_english  
**大小**: 约 1 GB  
**用途**: 多语言情感分类

#### 下载方式 A：从 Hugging Face 下载

```bash
# 使用 transformers 库下载
python -c "from transformers import AutoModelForSequenceClassification, AutoTokenizer; AutoModelForSequenceClassification.from_pretrained('xlm-roberta-base'); AutoTokenizer.from_pretrained('xlm-roberta-base')"
```

#### 下载方式 B：百度网盘

- **下载链接**: [待上传]
- **提取码**: xxxx
- **解压到**: `Chinese classification/emotion_chinese_english/`

---

### 3. 表情识别模型

**模型名称**: FER2013_VGG19  
**大小**: 约 76 MB  
**用途**: 面部表情识别（7种情绪）

#### 下载方式：百度网盘

- **下载链接**: [待上传]
- **提取码**: xxxx
- **文件**: `PrivateTest_model.t7`
- **放置位置**: `Expression Recognition/FER2013_VGG19/PrivateTest_model.t7`

---

### 4. dlib 人脸关键点检测模型

**模型名称**: shape_predictor_68_face_landmarks  
**大小**: 95 MB  
**用途**: 检测人脸 68 个关键点，用于 FACS 编码分析

#### 下载方式 A：官方下载（推荐）

1. 访问：http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
2. 下载并解压 `.bz2` 文件
3. 将 `shape_predictor_68_face_landmarks.dat` 放到 `Expression Recognition/` 目录

#### 下载方式 B：百度网盘

- **下载链接**: [待上传]
- **提取码**: xxxx
- **放置位置**: `Expression Recognition/shape_predictor_68_face_landmarks.dat`

---

## 🎯 可选模型（用于训练或实验）

### 5. 训练好的情感分类模型

如果你不想重新训练，可以直接使用我们训练好的模型：

- **模型**: `sentiment_model_0.6521025512302006.pth`
- **F1 Score**: 0.652
- **下载链接**: [待上传]
- **放置位置**: `Chinese classification/model/`

---

### 6. CK+ 数据集模型

用于 CK+ 数据集的表情识别：

- **模型**: `Test_model.t7`
- **准确率**: ~92%
- **下载链接**: [待上传]
- **放置位置**: `Expression Recognition/CK+_VGG19/`

---

## 📂 最终目录结构

下载完成后，你的目录结构应该如下：

```
DMSEmotion/
├── Chinese classification/
│   ├── bert-base-chinese/
│   │   ├── config.json ✅
│   │   ├── vocab.txt ✅
│   │   ├── tokenizer.json ✅
│   │   ├── tokenizer_config.json ✅
│   │   └── pytorch_model.bin ⬇️ 需要下载
│   │
│   ├── emotion_chinese_english/
│   │   ├── config.json ✅
│   │   ├── tokenizer_config.json ✅
│   │   ├── sentencepiece.bpe.model ✅
│   │   └── pytorch_model.bin ⬇️ 需要下载
│   │
│   └── model/
│       └── sentiment_model_*.pth ⬇️ 可选
│
└── Expression Recognition/
    ├── shape_predictor_68_face_landmarks.dat ⬇️ 需要下载
    │
    └── FER2013_VGG19/
        └── PrivateTest_model.t7 ⬇️ 需要下载
```

---

## ✅ 验证安装

下载完成后，运行以下命令验证模型是否正确安装：

```python
# 验证 BERT 模型
from transformers import BertTokenizer, BertForSequenceClassification
tokenizer = BertTokenizer.from_pretrained('Chinese classification/bert-base-chinese')
model = BertForSequenceClassification.from_pretrained('Chinese classification/bert-base-chinese')
print("✅ BERT 模型加载成功")

# 验证 dlib 模型
import dlib
predictor = dlib.shape_predictor('Expression Recognition/shape_predictor_68_face_landmarks.dat')
print("✅ dlib 模型加载成功")

# 验证表情识别模型
import torch
model = torch.load('Expression Recognition/FER2013_VGG19/PrivateTest_model.t7')
print("✅ 表情识别模型加载成功")
```

---

## 🔧 常见问题

### Q1: 下载速度太慢怎么办？

**A**: 
- 使用百度网盘下载（国内速度快）
- 使用 Hugging Face 镜像站：https://hf-mirror.com/
- 配置代理加速下载

### Q2: 模型文件损坏或无法加载？

**A**: 
- 重新下载模型文件
- 检查文件大小是否正确
- 确认文件路径是否正确

### Q3: 可以使用其他预训练模型吗？

**A**: 
可以！你可以使用任何兼容的模型：
- BERT 系列：`bert-base-chinese`, `chinese-bert-wwm-ext`
- RoBERTa 系列：`chinese-roberta-wwm-ext`
- 表情识别：任何 VGG/ResNet 训练的模型

只需修改代码中的模型路径即可。

---

## 📞 需要帮助？

如果下载或配置遇到问题，请：
1. 查看 [Issues](https://github.com/yourusername/DMSEmotion/issues)
2. 提交新的 Issue
3. 发送邮件至：[your-email@example.com]

---

## 🙏 模型来源致谢

- **BERT**: Google Research / Hugging Face
- **dlib**: Davis King (http://dlib.net/)
- **FER2013**: Kaggle Competition Dataset
- **CK+**: Carnegie Mellon University

---

**最后更新**: 2026-05-17
