from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from pathlib import Path

# Load the trained model and tokenizer from local paths
PROJECT_ROOT = Path(__file__).resolve().parent
model_path = PROJECT_ROOT / 'results' / 'checkpoint-86050'
tokenizer_path = PROJECT_ROOT / 'emotion_chinese_english'
tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
model = AutoModelForSequenceClassification.from_pretrained(str(model_path))

# 设置模型为评估模式，并禁用梯度计算
model.eval()

# Get the input text from the user with a prompt
input_text = input("请输入文本：")

# Tokenize the input text
encoded_input = tokenizer(
    input_text,
    padding=True,
    truncation=True,
    max_length=128,
    return_tensors='pt'
)

# Make predictions with no gradient calculation
with torch.no_grad():
    outputs = model(**encoded_input)
    predictions = torch.argmax(outputs.logits, dim=1)

label_map = {0: 'sadness', 1: 'happiness', 2: 'anger', 3: 'fear', 4: 'disgust'}

# Print the predicted label
print("预测情绪：", label_map[predictions.item()])
