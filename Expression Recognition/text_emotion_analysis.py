import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

class TextEmotionAnalyzer:
    def __init__(self, model_path, tokenizer_path):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model.eval()
        self.label_map = {0: '悲伤', 1: '快乐', 2: '愤怒', 3: '恐惧', 4: '厌恶'}

    def analyze_text_file(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件未找到: {file_path}")

        emotions = []
        emotion_labels = []
        sentences = []

        with open(file_path, 'r', encoding='utf-8') as file:
            lines = [line.strip() for line in file if line.strip()]

        for line in lines:
            encoded_input = self.tokenizer(
                line,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            )

            with torch.no_grad():
                outputs = self.model(**encoded_input)
                prediction = torch.argmax(outputs.logits, dim=1).item()

            emotion_label = self.label_map[prediction]

            emotions.append(prediction)
            emotion_labels.append(emotion_label)
            sentences.append(line)

        return sentences, emotions, emotion_labels
