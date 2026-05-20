# 🚀 快速开始指南

本指南将帮助你在 5 分钟内运行 DMSEmotion 项目。

---

## ⚡ 一键运行（推荐）

### Windows 用户

```powershell
# 1. 克隆项目
git clone https://github.com/yourusername/DMSEmotion.git
cd DMSEmotion

# 2. 安装依赖
pip install -r requirements.txt

# 3. 下载必需模型（自动）
python download_models.py

# 4. 运行程序
cd "Expression Recognition"
python app_gui2.py
```

### Linux/Mac 用户

```bash
# 1. 克隆项目
git clone https://github.com/yourusername/DMSEmotion.git
cd DMSEmotion

# 2. 安装依赖
pip3 install -r requirements.txt

# 3. 下载必需模型
python3 download_models.py

# 4. 运行程序
cd "Expression Recognition"
python3 app_gui2.py
```

---

## 📋 详细步骤

### 步骤 1: 环境准备

**检查 Python 版本**（需要 3.7+）：
```bash
python --version
```

**创建虚拟环境**（推荐）：
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

---

### 步骤 2: 安装依赖

```bash
pip install -r requirements.txt
```

**常见问题**：

- **dlib 安装失败**：
  ```bash
  # Windows: 下载预编译版本
  pip install dlib-19.22.0-cp39-cp39-win_amd64.whl
  
  # Linux: 安装依赖
  sudo apt-get install cmake
  sudo apt-get install libboost-all-dev
  pip install dlib
  ```

- **PyTorch 安装**：
  ```bash
  # CPU 版本
  pip install torch torchvision
  
  # GPU 版本（CUDA 11.1）
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu111
  ```

---

### 步骤 3: 下载模型

#### 方式 A: 自动下载（推荐）

运行自动下载脚本：
```bash
python download_models.py
```

#### 方式 B: 手动下载

查看 [MODEL_DOWNLOAD.md](MODEL_DOWNLOAD.md) 获取详细说明。

**最小配置**（仅下载必需模型）：
1. dlib 人脸模型（95MB）
2. BERT 模型（通过 transformers 自动下载）
3. 表情识别模型（76MB）

---

### 步骤 4: 配置讯飞语音 API（可选）

如果需要使用语音识别功能：

1. 注册讯飞开放平台账号：https://www.xfyun.cn/
2. 创建应用获取 API 密钥
3. 修改 `Expression Recognition/app_gui2.py`：

```python
self.speech_client = IatWsClient(
    appid='你的APPID',
    api_key='你的API_KEY',
    api_secret='你的API_SECRET',
    text_callback=debug_cb
)
```

**不配置的影响**：语音转文字功能无法使用，但不影响表情识别。

---

### 步骤 5: 运行程序

```bash
cd "Expression Recognition"
python app_gui2.py
```

程序启动后会看到主界面：
- 左侧：控制按钮和分析结果
- 右侧：情绪可视化图表

---

## 🎮 使用示例

### 示例 1: 单张图片识别

1. 点击 **"上传图片"** 按钮
2. 选择一张包含人脸的图片
3. 查看识别结果：
   - 情绪类别和置信度
   - FACS 编码分析
   - 情绪空间位置
   - 可视化图表

### 示例 2: 实时识别

1. 确保摄像头已连接
2. 点击 **"实时识别"** 按钮
3. 系统开始实时分析：
   - 面部表情
   - 语音转文字（如已配置）
   - 情绪变化趋势
4. 点击 **"保存数据"** 保存结果
5. 点击 **"数据分析"** 进行文本情感分析
6. 点击 **"结束识别"** 停止

---

## 🧪 测试安装

运行测试脚本验证安装：

```python
# test_installation.py
import sys

def test_imports():
    """测试所有依赖是否正确安装"""
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
    except:
        print("❌ PyTorch 未安装")
    
    try:
        import cv2
        print(f"✅ OpenCV {cv2.__version__}")
    except:
        print("❌ OpenCV 未安装")
    
    try:
        import dlib
        print("✅ dlib 已安装")
    except:
        print("❌ dlib 未安装")
    
    try:
        from transformers import BertTokenizer
        print("✅ Transformers 已安装")
    except:
        print("❌ Transformers 未安装")
    
    try:
        import tkinter
        print("✅ Tkinter 已安装")
    except:
        print("❌ Tkinter 未安装")

def test_models():
    """测试模型文件是否存在"""
    import os
    
    models = [
        ("dlib 模型", "Expression Recognition/shape_predictor_68_face_landmarks.dat"),
        ("表情识别模型", "Expression Recognition/FER2013_VGG19/PrivateTest_model.t7"),
        ("BERT 配置", "Chinese classification/bert-base-chinese/config.json"),
    ]
    
    for name, path in models:
        if os.path.exists(path):
            print(f"✅ {name}")
        else:
            print(f"❌ {name} - 文件不存在: {path}")

if __name__ == "__main__":
    print("=" * 50)
    print("测试依赖安装")
    print("=" * 50)
    test_imports()
    
    print("\n" + "=" * 50)
    print("测试模型文件")
    print("=" * 50)
    test_models()
```

运行测试：
```bash
python test_installation.py
```

---

## 🐛 常见问题

### Q1: 程序启动后摄像头无法打开

**解决方案**：
- 检查摄像头是否被其他程序占用
- 修改 `real_time_recognition.py` 中的摄像头索引：
  ```python
  self.cap = cv2.VideoCapture(0)  # 改为 1 或 2
  ```

### Q2: 识别速度很慢

**解决方案**：
- 安装 GPU 版本的 PyTorch
- 降低视频分辨率
- 减少处理帧率

### Q3: 中文显示乱码

**解决方案**：
- 确保系统安装了微软雅黑字体
- 修改 `app_gui2.py` 中的字体设置：
  ```python
  matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 或其他中文字体
  ```

### Q4: ModuleNotFoundError

**解决方案**：
```bash
# 重新安装缺失的模块
pip install [模块名]

# 或重新安装所有依赖
pip install -r requirements.txt --force-reinstall
```

---

## 📊 性能优化

### GPU 加速

如果有 NVIDIA 显卡：

1. 安装 CUDA Toolkit
2. 安装 GPU 版本的 PyTorch：
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```
3. 验证 GPU 可用：
   ```python
   import torch
   print(torch.cuda.is_available())  # 应该输出 True
   ```

### 内存优化

如果内存不足：
- 减小 batch_size
- 使用更小的模型
- 关闭不需要的可视化

---

## 📚 下一步

- 阅读 [README.md](README.md) 了解项目详情
- 查看 [MODEL_DOWNLOAD.md](MODEL_DOWNLOAD.md) 下载更多模型
- 浏览 `Chinese classification/ipynb/` 中的 Jupyter Notebook 示例
- 自定义训练模型（见 README 中的"自定义训练"部分）

---

## 💬 获取帮助

遇到问题？
- 查看 [Issues](https://github.com/yourusername/DMSEmotion/issues)
- 提交新的 Issue
- 发送邮件：[your-email@example.com]

---

**祝你使用愉快！** 🎉
