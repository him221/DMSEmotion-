#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
自动下载 DMSEmotion 项目所需的模型文件
"""

import os
import sys
import urllib.request
import bz2
import shutil
from pathlib import Path

def download_file(url, dest_path, description):
    """下载文件并显示进度"""
    print(f"\n📥 正在下载: {description}")
    print(f"   URL: {url}")
    print(f"   保存到: {dest_path}")
    
    try:
        def reporthook(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size)
            sys.stdout.write(f"\r   进度: {percent}% ")
            sys.stdout.flush()
        
        urllib.request.urlretrieve(url, dest_path, reporthook)
        print("\n   ✅ 下载完成")
        return True
    except Exception as e:
        print(f"\n   ❌ 下载失败: {e}")
        return False

def decompress_bz2(src_path, dest_path):
    """解压 .bz2 文件"""
    print(f"\n📦 正在解压: {src_path}")
    try:
        with bz2.open(src_path, 'rb') as src, open(dest_path, 'wb') as dest:
            shutil.copyfileobj(src, dest)
        print("   ✅ 解压完成")
        os.remove(src_path)  # 删除压缩文件
        return True
    except Exception as e:
        print(f"   ❌ 解压失败: {e}")
        return False

def download_dlib_model():
    """下载 dlib 人脸关键点检测模型"""
    url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    dest_dir = Path("Expression Recognition")
    dest_dir.mkdir(exist_ok=True)
    
    compressed_path = dest_dir / "shape_predictor_68_face_landmarks.dat.bz2"
    final_path = dest_dir / "shape_predictor_68_face_landmarks.dat"
    
    if final_path.exists():
        print(f"\n✅ dlib 模型已存在: {final_path}")
        return True
    
    if download_file(url, compressed_path, "dlib 人脸关键点模型 (95MB)"):
        return decompress_bz2(compressed_path, final_path)
    return False

def download_bert_model():
    """使用 transformers 下载 BERT 模型"""
    print("\n📥 正在下载: BERT 中文预训练模型")
    print("   这可能需要几分钟...")
    
    try:
        from transformers import BertTokenizer, BertForSequenceClassification
        
        model_name = "bert-base-chinese"
        save_dir = Path("Chinese classification/bert-base-chinese")
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 检查是否已存在
        if (save_dir / "pytorch_model.bin").exists():
            print("   ✅ BERT 模型已存在")
            return True
        
        print("   正在下载 tokenizer...")
        tokenizer = BertTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(save_dir)
        
        print("   正在下载模型...")
        model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=5,
            ignore_mismatched_sizes=True
        )
        model.save_pretrained(save_dir)
        
        print("   ✅ BERT 模型下载完成")
        return True
    except Exception as e:
        print(f"   ❌ BERT 模型下载失败: {e}")
        print("   提示: 请确保已安装 transformers 库")
        print("   运行: pip install transformers")
        return False

def download_emotion_model():
    """下载情感分类模型"""
    print("\n📥 正在下载: XLM-RoBERTa 情感分类模型")
    print("   这可能需要几分钟...")
    
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        
        model_name = "xlm-roberta-base"
        save_dir = Path("Chinese classification/emotion_chinese_english")
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 检查是否已存在
        if (save_dir / "pytorch_model.bin").exists():
            print("   ✅ 情感分类模型已存在")
            return True
        
        print("   正在下载 tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(save_dir)
        
        print("   正在下载模型...")
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.save_pretrained(save_dir)
        
        print("   ✅ 情感分类模型下载完成")
        return True
    except Exception as e:
        print(f"   ❌ 情感分类模型下载失败: {e}")
        print("   提示: 请确保已安装 transformers 库")
        return False

def create_model_directories():
    """创建必要的模型目录"""
    directories = [
        "Expression Recognition/FER2013_VGG19",
        "Expression Recognition/CK+_VGG19",
        "Chinese classification/model",
        "Chinese classification/results",
        "Data Recording/csv-data",
        "Data Recording/json-data",
        "Data Recording/png-data",
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("\n✅ 已创建必要的目录结构")

def create_readme_for_manual_download():
    """为需要手动下载的模型创建说明文件"""
    fer_readme = """# 表情识别模型

由于文件较大，请手动下载表情识别模型。

## 下载方式

### 方式 1: 百度网盘
- 下载链接: [待上传]
- 提取码: xxxx
- 文件名: PrivateTest_model.t7 (76MB)

### 方式 2: 自行训练
运行以下命令训练模型:
```bash
cd "Expression Recognition"
python mainpro_FER.py
```

## 放置位置
将下载的 `PrivateTest_model.t7` 文件放到当前目录。

## 验证
运行以下代码验证模型是否正确:
```python
import torch
model = torch.load('PrivateTest_model.t7')
print("模型加载成功!")
```
"""
    
    fer_path = Path("Expression Recognition/FER2013_VGG19/README.md")
    if not fer_path.exists():
        with open(fer_path, 'w', encoding='utf-8') as f:
            f.write(fer_readme)
        print(f"\n✅ 已创建说明文件: {fer_path}")

def main():
    """主函数"""
    print("=" * 60)
    print("  DMSEmotion 模型自动下载工具")
    print("=" * 60)
    print("\n本工具将自动下载以下模型:")
    print("  1. dlib 人脸关键点模型 (95MB)")
    print("  2. BERT 中文预训练模型 (~400MB)")
    print("  3. XLM-RoBERTa 情感分类模型 (~1GB)")
    print("\n⚠️  注意: 下载可能需要 10-30 分钟，取决于网络速度")
    print("⚠️  如果下载失败，请查看 MODEL_DOWNLOAD.md 手动下载")
    
    response = input("\n是否继续? (y/n): ")
    if response.lower() != 'y':
        print("已取消")
        return
    
    # 创建目录结构
    create_model_directories()
    
    # 下载模型
    results = []
    
    print("\n" + "=" * 60)
    print("开始下载模型...")
    print("=" * 60)
    
    # 1. dlib 模型
    results.append(("dlib 人脸模型", download_dlib_model()))
    
    # 2. BERT 模型
    results.append(("BERT 模型", download_bert_model()))
    
    # 3. 情感分类模型
    results.append(("情感分类模型", download_emotion_model()))
    
    # 创建手动下载说明
    create_readme_for_manual_download()
    
    # 显示结果
    print("\n" + "=" * 60)
    print("下载结果汇总")
    print("=" * 60)
    
    for name, success in results:
        status = "✅ 成功" if success else "❌ 失败"
        print(f"{status} - {name}")
    
    print("\n" + "=" * 60)
    
    if all(success for _, success in results):
        print("🎉 所有模型下载完成!")
        print("\n下一步:")
        print("  1. 手动下载表情识别模型 (见 MODEL_DOWNLOAD.md)")
        print("  2. 运行程序: cd 'Expression Recognition' && python app_gui2.py")
    else:
        print("⚠️  部分模型下载失败")
        print("\n请查看 MODEL_DOWNLOAD.md 获取手动下载说明")
    
    print("=" * 60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断下载")
    except Exception as e:
        print(f"\n\n❌ 发生错误: {e}")
        print("请查看 MODEL_DOWNLOAD.md 手动下载模型")
