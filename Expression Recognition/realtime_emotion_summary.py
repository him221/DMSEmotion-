# realtime_emotion_summary.py
from collections import Counter

class RealTimeEmotionSummary:
    def __init__(self):
        self.emotion_list = []

    def record_emotion(self, predicted_class):
        self.emotion_list.append(predicted_class)

    def summarize_emotions(self):
        if not self.emotion_list:
            return "本次识别未检测到有效情绪。"

        total_emotions = len(self.emotion_list)
        emotion_counter = Counter(self.emotion_list)

        most_common_emotion, count = emotion_counter.most_common(1)[0]

        # 构建情绪占比信息
        summary_text = f"情绪识别总结：\n最主要情绪为：【{most_common_emotion}】（占比 {count / total_emotions * 100:.2f}%）\n\n详细情绪占比:\n"
        for emotion, emotion_count in emotion_counter.items():
            percentage = (emotion_count / total_emotions) * 100
            summary_text += f"- {emotion}: {percentage:.2f}%\n"

        return summary_text

    def reset_summary(self):
        self.emotion_list.clear()

