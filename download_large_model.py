#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
下载大型情感分类模型
用于 Git LFS 精简方案
"""

import os
import sys
from pathlib import Path

def download_emotion_model():
    """下载 XLM-RoBERTa 情感分类模型"""
    print("=" * 60)
    print("  下载大型情感分类模型")
    print("=" * 60)
    print()
    print("正在下载 XLM-RoBERTa 情感分类模型...")
    print("大小: 约 1GB")
    print("这可能需要几分钟，请耐心等待...")
    print()
    
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        
        model_name = "xlm-roberta-base"
        save_dir = Path("Chinese classification/emotion_chinese_english")
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 检查是否已存在
        if (save_dir / "pytorch_model.bin").exists():
            print("✅ 模型已存在，无需重新下载")
            return True
        
        print("📥 正在下载 tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(save_dir)
        print("   ✓ Tokenizer 下载完成")
        
        print("📥 正在下载模型（这是最大的文件）...")
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.save_pretrained(save_dir)
        print("   ✓ 模型下载完成")
        
        print()
        print("=" * 60)
        print("🎉 下载完成！")
        print("=" * 60)
        print()
        print("现在可以运行程序了：")
        print('  cd "Expression Recognition"')
        print("  python app_gui2.py")
        print()
        
        return True
        
    except ImportError:
        print("❌ 错误: 未安装 transformers 库")
        print()
        print("请先安装依赖:")
        print("  pip install transformers")
        return False
        
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        print()
        print("可能的原因:")
        print("  1. 网络连接问题")
        print("  2. 磁盘空间不足")
        print("  3. 防火墙阻止")
        print()
        print("解决方案:")
        print("  1. 检查网络连接")
        print("  2. 使用代理或 VPN")
        print("  3. 手动下载（见 MODEL_DOWNLOAD.md）")
        return False

def main():
    """主函数"""
    # 检查是否在项目根目录
    if not os.path.exists("Expression Recognition"):
        print("❌ 错误: 请在项目根目录运行此脚本")
        print()
        print("正确的运行方式:")
        print("  cd DMSEmotion")
        print("  python download_large_model.py")
        sys.exit(1)
    
    success = download_emotion_model()
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断下载")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ 发生错误: {e}")
        sys.exit(1)
