# multimodal_fusion.py
from collections import Counter
from typing import List, Dict

def summarize_face_emotion_distribution(face_emotions: List[str]) -> Dict[str, float]:
    """
    统计人脸情绪的分布占比（返回百分比形式的字典）
    """
    if not face_emotions:
        return {}
    total = len(face_emotions)
    counts = Counter(face_emotions)
    return {emotion: round(count / total * 100, 2) for emotion, count in counts.items()}

def summarize_text_emotion_distribution(text_emotions: List[str]) -> Dict[str, float]:
    """
    统计文本情绪的分布占比（返回百分比形式的字典）
    """
    if not text_emotions:
        return {}
    total = len(text_emotions)
    counts = Counter(text_emotions)
    return {emotion: round(count / total * 100, 2) for emotion, count in counts.items()}

def fuse_emotions(face_distribution: Dict[str, float], text_distribution: Dict[str, float]) -> str:
    """
    多模态融合策略：
    - 如果文本情绪的最大占比大于人脸最大占比，则以文本主导
    - 否则以人脸主导
    - 如果均为空，则返回“未知”
    """
    if not face_distribution and not text_distribution:
        return "未知"

    face_sorted = sorted(face_distribution.items(), key=lambda x: x[1], reverse=True)
    text_sorted = sorted(text_distribution.items(), key=lambda x: x[1], reverse=True)

    face_top_emotion, face_top_value = face_sorted[0] if face_sorted else ("未知", 0)
    text_top_emotion, text_top_value = text_sorted[0] if text_sorted else ("未知", 0)

    return text_top_emotion if text_top_value >= face_top_value else face_top_emotion

def fusion_summary(face_distribution: Dict[str, float],
                   text_distribution: Dict[str, float],
                   fused_emotion: str) -> str:
    """
    整理展示文本：人脸情绪分布 + 文本情绪分布 + 综合结果
    """
    if not face_distribution:
        face_summary = "未知"
    else:
        face_sorted = sorted(face_distribution.items(), key=lambda x: x[1], reverse=True)
        face_summary = '，'.join([f"{emotion} {percent:.1f}%" for emotion, percent in face_sorted])

    if not text_distribution:
        text_summary = "未知"
    else:
        text_sorted = sorted(text_distribution.items(), key=lambda x: x[1], reverse=True)
        text_summary = '，'.join([f"{emotion} {percent:.1f}%" for emotion, percent in text_sorted])

    return (f"多模态情绪融合结果：\n"
            f"---- 人脸识别情绪分布：{face_summary}\n"
            f"---- 文本识别情绪分布：{text_summary}\n"
            f"----→ 综合多模态识别结果：----{fused_emotion}----占比最大!")
