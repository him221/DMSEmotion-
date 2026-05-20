# dynamic_barchart.py
import time
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
# class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
# color_list = ['red', 'orangered', 'darkorange', 'limegreen', 'darkgreen', 'royalblue', 'navy']
# color_map = dict(zip(class_names, color_list))

color_map = {
        'Angry': 'red',
        'Disgust': 'orangered',
        'Fear': 'darkorange',
        'Happy': 'limegreen',
        'Sad': 'darkgreen',
        'Surprise': 'royalblue',
        'Neutral': 'navy'
    }

class EmotionBarChart:
    """
    记录最近5分钟的表情数据，并在指定Canvas上绘制直方图，展示各表情出现次数(或占比)。
    """
    def __init__(self, canvas, class_names, max_duration=300):
        """
        :param canvas: 要绘图的Tkinter Canvas（如app_gui.py里第二个方框）
        :param class_names: 表情类别列表, e.g. ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']
        :param max_duration: 数据保留时长(秒)，默认5分钟=300秒
        """
        self.canvas = canvas
        self.class_names = class_names
        self.max_duration = max_duration
        self.start_time = time.time()

        # 存储 (timestamp, predicted_class)
        self.emotion_history = []

    def add_emotion(self, predicted_class):
        """
        每次有新的表情识别结果时调用，记录到 emotion_history
        """
        current_time = time.time() - self.start_time
        self.emotion_history.append((current_time, predicted_class))

        # 移除超过5分钟的数据
        while self.emotion_history and (current_time - self.emotion_history[0][0] > self.max_duration):
            self.emotion_history.pop(0)

    def update_chart(self):
        """
        在 self.canvas 上绘制直方图，展示最近5分钟各表情的出现次数
        """
        # 清空Canvas上的旧图表
        for widget in self.canvas.winfo_children():
            widget.destroy()

        if not self.emotion_history:
            return

        # 1) 计算计数
        counts = {emo: 0 for emo in self.class_names}
        for (t, cls) in self.emotion_history:
            if cls in counts:
                counts[cls] += 1

        # 2) 准备X轴(表情类别)和Y轴(计数)
        labels = list(counts.keys())
        values = list(counts.values())

        # 3) 绘制直方图
        fig = Figure(figsize=(3, 3))
        ax = fig.add_subplot(111)
        # 2) 构造与labels顺序相对应的颜色列表
        bar_colors = [color_map[emo] for emo in labels]
        ax.bar(labels, values, color=bar_colors)
        # ax.set_title("5分钟内表情占比")
        ax.set_xlabel("表情类别")
        ax.set_ylabel("出现次数")

        # x轴标签旋转一下，避免重叠
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

        fig.tight_layout()

        # 4) 在Tkinter Canvas上嵌入
        canvas_bar = FigureCanvasTkAgg(fig, master=self.canvas)
        canvas_bar.draw()
        canvas_bar.get_tk_widget().pack(expand=True, fill='both')
