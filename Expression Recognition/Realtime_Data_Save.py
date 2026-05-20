import os
import csv
import json
import matplotlib.pyplot as plt
from datetime import datetime


class DataRecorder:
    def __init__(self):
        self.emotion_log = []
        self.base_dir = 'Data Recording'
        self.csv_dir = os.path.join(self.base_dir, 'csv-data')
        self.json_dir = os.path.join(self.base_dir, 'json-data')
        self.png_dir = os.path.join(self.base_dir, 'png-data')
        self.ensure_directories()

    def ensure_directories(self):
        for directory in [self.csv_dir, self.json_dir, self.png_dir]:
            os.makedirs(directory, exist_ok=True)

    def log_emotion(self, timestamp, emotion, probabilities):
        self.emotion_log.append({
            'timestamp': timestamp,
            'emotion': emotion,
            'probabilities': probabilities
        })

    def save_data(self):
        if not self.emotion_log:
            return

        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        save_path_csv = os.path.join(self.csv_dir, timestamp)
        save_path_json = os.path.join(self.json_dir, timestamp)
        save_path_png = os.path.join(self.png_dir, timestamp)

        os.makedirs(save_path_csv, exist_ok=True)
        os.makedirs(save_path_json, exist_ok=True)
        os.makedirs(save_path_png, exist_ok=True)

        # Save CSV
        csv_path = os.path.join(save_path_csv, f'{timestamp}.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['时间戳', '情绪', '概率'])
            for entry in self.emotion_log:
                writer.writerow([entry['timestamp'], entry['emotion'], entry['probabilities'][entry['emotion']]])

        # Save JSON
        json_path = os.path.join(save_path_json, f'{timestamp}.json')
        json_data = {entry['timestamp']: {
            'emotion': entry['emotion'],
            'probabilities': {emotion: float(prob) for emotion, prob in entry['probabilities'].items()}
        } for entry in self.emotion_log}

        with open(json_path, 'w', encoding='utf-8') as jsonfile:
            json.dump(json_data, jsonfile, ensure_ascii=False, indent=4)

        # Save PNG
        self.save_emotion_trends(save_path_png, timestamp)

    def save_emotion_trends(self, save_path_png, timestamp):
        times = [entry['timestamp'] for entry in self.emotion_log]
        emotions = list(self.emotion_log[0]['probabilities'].keys())

        # Emotion trends
        plt.figure(figsize=(12, 6))
        for emotion in emotions:
            probs = [entry['probabilities'][emotion] for entry in self.emotion_log]
            plt.plot(times, probs, label=emotion)

        plt.xticks(rotation=45, ha='right')
        plt.xlabel('时间')
        plt.ylabel('概率')
        plt.title('情绪概率随时间变化的趋势')
        plt.legend(loc='upper right')  # 将图例放置于右上角
        plt.tight_layout()

        trend_path = os.path.join(save_path_png, f'{timestamp}_trends.png')
        plt.savefig(trend_path)
        plt.close()

        # Emotion proportion
        emotion_counts = {emotion: 0 for emotion in emotions}
        for entry in self.emotion_log:
            emotion_counts[entry['emotion']] += 1

        plt.figure(figsize=(10, 5))
        plt.bar(emotion_counts.keys(), emotion_counts.values(), color='skyblue')
        plt.xlabel('情绪序列')
        plt.ylabel('次数')
        plt.title('情感在识别中的频率')
        plt.tight_layout()

        count_path = os.path.join(save_path_png, f'{timestamp}_counts.png')
        plt.savefig(count_path)
        plt.close()
