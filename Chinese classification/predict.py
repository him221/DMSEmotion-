from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


LABEL_MAP = {0: 'sadness', 1: 'happiness', 2: 'anger', 3: 'fear', 4: 'disgust'}


def predict_by_prompt(transformer, data):
    model = AutoModelForSequenceClassification.from_pretrained(transformer, ignore_mismatched_sizes=True)
    tokenizer = AutoTokenizer.from_pretrained(transformer)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    text_list = data['content'].tolist()
    label_list = data['label'].tolist()
    unique_labels = sorted(list(set(label_list)))
    y_pred = []
    y_true = []

    for index, text in enumerate(tqdm(text_list)):
        probs = []
        with torch.no_grad():
            for label in unique_labels:
                encoded = tokenizer.encode(text, label, return_tensors='pt', truncation_strategy='do_not_truncate')
                encoded = encoded.to(device)
                logits = model(encoded)[0]
                entail_contradiction_logits = logits[:, [0, 1]]
                prob_label_is_true = entail_contradiction_logits.softmax(dim=1)[:, 1]
                probs.append(prob_label_is_true.detach().cpu().numpy()[0])
        y_pred.append(unique_labels[np.argmax(np.array(probs))])
        y_true.append(label_list[index])

    return pd.DataFrame({'text': text_list, 'True': y_true, 'predict': y_pred})


def predict_by_roberta(tokenizer, model, data):
    text_list = data['content'].tolist()
    label_list = data['label'].tolist()
    pre_label = []

    for input_text in text_list:
        encoded_input = tokenizer(input_text, padding=True, truncation=True, max_length=128, return_tensors='pt')
        outputs = model(**encoded_input)
        predictions = torch.argmax(outputs.logits, dim=1)
        pre_label.append(LABEL_MAP[predictions.item()])

    return pd.DataFrame({'text': text_list, 'True': label_list, 'predict': pre_label})


def predict_by_bert(tokenizer, model, data, device):
    text_list = data['content'].tolist()
    label_list = data['label'].tolist()
    pre_label = []

    for text in text_list:
        inputs = tokenizer(text, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)
            pre_label.append(LABEL_MAP[predictions.item()])

    return pd.DataFrame({'text': text_list, 'True': label_list, 'predict': pre_label})


if __name__ == '__main__':
    project_root = Path(__file__).resolve().parent
    data_path = project_root / 'data' / '天池比赛情绪分类训练数据集.xlsx'
    data = pd.read_excel(data_path)
    data.set_index('index', inplace=True)
    print(data['label'].unique())

    transformer_path = str(project_root / 'emotion_chinese_english')
    result = predict_by_prompt(transformer_path, data[:15])
    result.to_csv(project_root / 'text.csv', index=False)
