# dynamic_linechart.py

import time
import matplotlib
matplotlib.use('TkAgg')  # 若需Tkinter界面绘制
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

class EmotionLineChart:
    """
    负责记录最近5分钟的表情数据，并在指定Canvas上绘制动态折线图。
    """

    def __init__(self, canvas, class_names, max_duration=300):
        """
        :param canvas: 要绘图的Tkinter Canvas（如app_gui.py里右侧第一个方框）
        :param class_names: 你的表情类别列表，比如 ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        :param max_duration: 数据保留时长(秒)，默认5分钟=300秒
        """
        self.canvas = canvas
        self.class_names = class_names
        self.max_duration = max_duration

        # 记录开始时间，用于计算相对时间戳
        self.start_time = time.time()

        # 存储 (timestamp, predicted_class)
        self.emotion_history = []

        # 映射表情 -> 数字(0,1,2,...)
        self.emotion_to_num = {emo: i for i, emo in enumerate(class_names)}

    def add_emotion(self, predicted_class):
        """
        每次有新的表情识别结果时调用，记录到emotion_history里
        """
        current_time = time.time() - self.start_time
        self.emotion_history.append((current_time, predicted_class))

        # 移除超过5分钟的数据
        while self.emotion_history and (current_time - self.emotion_history[0][0] > self.max_duration):
            self.emotion_history.pop(0)

    def update_chart(self):
        # 先清空Canvas上的旧图表
        for widget in self.canvas.winfo_children():
            widget.destroy()

        if not self.emotion_history:
            return

        times = []
        values = []
        for (t, cls) in self.emotion_history:
            times.append(t)
            # 将表情类别映射为数字 0~6
            values.append(self.emotion_to_num.get(cls, 6))

        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        fig = Figure(figsize=(3, 3))
        ax = fig.add_subplot(111)

        # === 使用阶梯图 (step) ===
        ax.step(times, values,
                where='post',  # 'post'表示在该点之后开始台阶，也可试试'pre'/'mid'
                linewidth=1,  # 线条宽度较细
                color='blue')

        # === 可选：再绘制散点来标识具体时间点 ===
        # s=20表示散点大小(可再调小/调大)
        ax.scatter(times, values, s=20, c='blue', alpha=0.8)

        # 设置标题、坐标轴
        ax.set_title("5分钟内情绪变化(阶梯图)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("表情索引")

        # 设置y轴ticks为情绪名称
        ax.set_yticks(range(len(self.class_names)))
        ax.set_yticklabels(self.class_names)

        # x轴范围只显示最近 max_duration 秒
        current_time = times[-1]
        ax.set_xlim([max(0, current_time - self.max_duration), current_time])

        # 给y轴留点空隙
        ax.set_ylim([-0.5, len(self.class_names) - 0.5])

        fig.tight_layout()

        canvas_chart = FigureCanvasTkAgg(fig, master=self.canvas)
        canvas_chart.draw()
        canvas_chart.get_tk_widget().pack(expand=True, fill='both')
