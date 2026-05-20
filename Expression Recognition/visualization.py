# visualization.py
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from PIL import Image, ImageTk

def draw_radar(radar_canvas, emotion_categories, emotion_values):
    """
    在 radar_canvas 上绘制情绪轮雷达图
    """
    # 清空旧的雷达图内容和所有子组件
    for widget in radar_canvas.winfo_children():
        widget.destroy()

    fig_radar = Figure(figsize=(3, 3))
    ax_radar = fig_radar.add_subplot(111, polar=True)
    angles = np.linspace(0, 2 * np.pi, len(emotion_categories), endpoint=False).tolist()
    angles += angles[:1]  # 闭合角度
    emotion_values += emotion_values[:1]  # 闭合图形

    ax_radar.fill(angles, emotion_values, color='skyblue', alpha=0.5)
    ax_radar.plot(angles, emotion_values, color='blue', linewidth=2)
    ax_radar.set_yticklabels([])
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(emotion_categories, fontsize=10)

    canvas_radar = FigureCanvasTkAgg(fig_radar, master=radar_canvas)
    canvas_radar.draw()
    canvas_radar.get_tk_widget().pack(expand=True, fill='both')


def draw_probability(prob_canvas, class_names, scores, color_list):
    # 1) 清空旧图表
    for widget in prob_canvas.winfo_children():
        widget.destroy()

    # 2) 创建 Figure 和子图
    fig_prob = Figure(figsize=(4, 2))
    ax_prob = fig_prob.add_subplot(111)

    # 3) 生成x轴位置（与class_names一一对应）
    x_positions = range(len(class_names))  # 或者用 np.arange(len(class_names))

    # 4) 绘制柱状图，x轴用整数序列
    ax_prob.bar(x_positions, scores, color=color_list)

    # 5) 手动设定x轴刻度
    ax_prob.set_xticks(x_positions)
    # 6) 再设置刻度标签
    ax_prob.set_xticklabels(class_names, rotation=45, fontsize=8)

    # 7) 布局调整
    fig_prob.tight_layout()

    # 8) 将绘图嵌入Tkinter的Canvas
    canvas_prob = FigureCanvasTkAgg(fig_prob, master=prob_canvas)
    canvas_prob.draw()
    canvas_prob.get_tk_widget().pack(expand=True, fill='both')

def draw_va(va_canvas, va_dict, scores, class_names, color_list, valence_weighted, arousal_weighted):
    """
    绘制情绪效价-唤醒度散点图
    """
    for widget in va_canvas.winfo_children():
        widget.destroy()

    fig_va = Figure(figsize=(4, 2.5))
    ax_va = fig_va.add_subplot(111)
    for emo, (val, aro) in va_dict.items():
        idx = class_names.index(emo)
        ax_va.scatter(val, aro, c=color_list[idx], s=50)
        ax_va.text(val, aro + 0.07, emo, fontsize=8, ha='center')

    ax_va.scatter(valence_weighted, arousal_weighted, c='magenta', s=150, marker='*',
                  edgecolors='black', label='当前情绪(综合位置)')
    ax_va.axhline(0, color='grey', linewidth=0.5, linestyle='--')
    ax_va.axvline(0, color='grey', linewidth=0.5, linestyle='--')
    ax_va.set_xlim(-1, 1)
    ax_va.set_ylim(-1, 1)
    ax_va.set_xlabel('效价', fontsize=9)
    ax_va.set_ylabel('唤醒度', fontsize=9)
    fig_va.tight_layout()

    canvas_va = FigureCanvasTkAgg(fig_va, master=va_canvas)
    canvas_va.draw()
    canvas_va.get_tk_widget().pack(expand=True, fill='both')


def draw_face(image_canvas, raw_img, landmarks):
    """
    显示输入图片并在上面画出关键点
    """
    fig_face = Figure(figsize=(5, 4))
    ax_face = fig_face.add_subplot(111)
    ax_face.imshow(raw_img)
    ax_face.axis('off')
    for (x, y) in landmarks:
        ax_face.plot(x, y, 'go', markersize=2)

    canvas_face = FigureCanvasTkAgg(fig_face, master=image_canvas)
    canvas_face.draw()
    canvas_face.get_tk_widget().pack(expand=True, fill='both')
    return canvas_face


def draw_emoji(emoji_canvas, predicted_class):
    """
    在 emoji_canvas 上显示对应表情的 emoji 图片
    """
    emoji_canvas.delete('all')
    try:
        emoji_img = Image.open(f'images/emojis/{predicted_class}.png').resize((80, 80))
        emoji_photo = ImageTk.PhotoImage(emoji_img)
        emoji_canvas.create_image(40, 40, image=emoji_photo, anchor='center')
        emoji_canvas.image = emoji_photo

        # # 保留你原本重复的显示逻辑
        # emoji_photo2 = ImageTk.PhotoImage(emoji_img)
        # emoji_canvas.create_image(80, 60, image=emoji_photo2, anchor='center')
        # emoji_canvas.image2 = emoji_photo2
    except FileNotFoundError:
        pass  # 如果没有对应图片，可做一个容错处理
