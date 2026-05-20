# dynamic_textlog.py

import time
import datetime
import tkinter as tk

class TimeLog:
    """
    记录最近5分钟内的表情识别结果，并在一个带滚动条的Text控件中显示。
    - 最新日志追加在末尾
    - 不自动滚动到底部（保持顶部显示）
    - 仅保留最近max_duration秒的数据
    """

    def __init__(self, parent, max_duration=300):
        """
        :param parent: 第三个方框(va_frame) 作为日志的容器
        :param max_duration: 保留数据时长(秒)，默认5分钟=300秒
        """
        self.parent = parent
        self.max_duration = max_duration
        self.emotion_history = []  # [(timestamp, predicted_class)]

        # 创建子frame来放置 Text + Scrollbar
        self.frame = tk.Frame(self.parent, bg='#FFFFFF')
        # 让子frame占满父容器
        self.frame.pack(expand=True, fill='both')

        # 垂直滚动条
        self.scrollbar = tk.Scrollbar(self.frame, orient=tk.VERTICAL)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Text控件
        self.text_widget = tk.Text(
            self.frame,
            width=30,
            height=10,
            yscrollcommand=self.scrollbar.set
        )
        self.text_widget.pack(side=tk.LEFT, expand=True, fill='both')

        # 关联滚动条
        self.scrollbar.config(command=self.text_widget.yview)

    def add_emotion(self, predicted_class):
        """每次识别到表情时调用，记录日志数据"""
        current_time = time.time()
        self.emotion_history.append((current_time, predicted_class))

        # 移除超过5分钟的数据
        while self.emotion_history and (current_time - self.emotion_history[0][0] > self.max_duration):
            self.emotion_history.pop(0)

    def update_log(self):
        """
        刷新Text控件，展示最近5分钟的数据 (从旧到新，最新在末尾)
        不自动滚动到末尾，保持顶部显示
        """
        # 1) 清空Text
        self.text_widget.delete('1.0', tk.END)

        # 2) 依次插入
        for (t, emo) in self.emotion_history:
            dt = datetime.datetime.fromtimestamp(t)
            # 使用微秒区分同一秒多次识别
            time_str = dt.strftime('%H:%M:%S.%f')
            line = f"[{time_str}] {emo}\n"
            self.text_widget.insert(tk.END, line)

        # 3) 强制视图移动到顶部
        self.text_widget.yview_moveto(0.0)
