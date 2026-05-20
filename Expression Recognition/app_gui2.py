import matplotlib
matplotlib.use('TkAgg')
import tkinter as tk
from tkinter import filedialog, messagebox
import dlib
import cv2
import threading
import glob
import matplotlib.pyplot as plt
import os
import queue
from pathlib import Path
from PIL import Image, ImageTk

# ======== 从你已有的模块中导入 ========
from real_time_recognition import RealTimeRecognizer
from iat_ws_client import IatWsClient
from dynamic_linechart import EmotionLineChart
from dynamic_barchart import EmotionBarChart
from dynamic_textlog import TimeLog
from text_emotion_analysis import TextEmotionAnalyzer
from Realtime_Data_Save import DataRecorder
from image_recognition import predict_expression, class_names
from facs_analysis import get_facs_from_landmarks, facs_dict
from visualization import (
    draw_radar, draw_probability, draw_va, draw_face, draw_emoji
)
from realtime_emotion_summary import RealTimeEmotionSummary
from multimodal_fusion import (
    fuse_emotions,
    fusion_summary,
    summarize_face_emotion_distribution,
    summarize_text_emotion_distribution
)

# ======== 中文字体支持 ========
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

color_list = ['red', 'orangered', 'darkorange', 'limegreen', 'darkgreen', 'royalblue', 'navy']

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEXT_MODEL_PATH = PROJECT_ROOT / 'Chinese classification' / 'results' / 'checkpoint-86050'
TEXT_TOKENIZER_PATH = PROJECT_ROOT / 'Chinese classification' / 'emotion_chinese_english'

class EmotionRecognitionApp:
    def __init__(self, root):
        """
            界面:
            - 左侧: 按钮 + 分析结果(上), emoji区域(中,横向布局), 图片/视频(下)
            - 右侧三行: (雷达图 / 直方图 / 第三个方框)
            - 图表标题随模式切换
            - 不影响原功能(单张图片 / 实时识别)
        """
        self.root = root
        self.root.title('DMSEmotion——基于多模态的驾乘人员情绪监测系统')
        self.root.geometry('1200x930')
        self.root.configure(bg='#FAFAFA')

        self.current_mode = 'none'
        self.data_recorder = DataRecorder()

        # 在EmotionRecognitionApp.__init__方法中加载情绪分析模型
        model_path = str(TEXT_MODEL_PATH)
        tokenizer_path = str(TEXT_TOKENIZER_PATH)
        self.text_emotion_analyzer = TextEmotionAnalyzer(model_path, tokenizer_path)

        # 中文情绪类别
        self.emotion_categories = ['快乐', '愤怒', '悲伤', '恐惧', '厌恶', '惊讶', '中性']
        self.emotion_values = [0] * len(self.emotion_categories)
        # 实时情绪总结模块
        self.emotion_summary = RealTimeEmotionSummary()

        # ========   GUI布局    ========
        # ======== 主布局(左右) ========
        main_frame = tk.Frame(self.root, bg='#FAFAFA')
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        main_frame.columnconfigure(0, weight=3)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)

        # ======== 左侧(按钮+结果 / emoji / 图片) ========
        left_frame = tk.Frame(main_frame, bg='#FFF8E1', bd=2, relief='groove')
        left_frame.grid(row=0, column=0, sticky='nsew', padx=(0, 5))
        left_frame.rowconfigure(0, minsize=180)  # 顶部区域至少 180 像素
        left_frame.rowconfigure(1, minsize=130)
        left_frame.rowconfigure(2, weight=1)  # 剩余空间全部给底部

        # ======== 右侧(三行图表) ========
        right_frame = tk.Frame(main_frame, bg='#F1F8E9', bd=2, relief='groove')
        right_frame.grid(row=0, column=1, sticky='nsew', padx=(5, 0))
        right_frame.rowconfigure([0, 1, 2], weight=1)

        # ========== 左上(按钮 + 分析结果) ==========
        left_top_frame = tk.Frame(left_frame, bg='#FFFDE7', bd=1, relief='ridge')
        left_top_frame.grid(row=0, column=0, sticky='nsew', pady=5, padx=5)

        tk.Label(left_top_frame, text="驾乘人员情绪监测系统DMSEmotion",
                 font=("Microsoft YaHei", 18, 'bold'),
                 bg='#FFFDE7', fg="#333").pack(anchor='nw', pady=(10, 5), padx=10)

        tk.Label(left_top_frame, text="可进行单张图片识别或实时识别",
                 font=("Microsoft YaHei", 12), bg='#FFFDE7', fg="#666") \
            .pack(anchor='nw', pady=(0, 10), padx=10)

        # 按钮样式
        btn_style = dict(font=("Microsoft YaHei", 12, 'bold'), fg='white', width=15, relief='raised')
        self.upload_button = tk.Button(left_top_frame, text="上传图片",
                                       command=self.start_image_mode,
                                       bg='#F57C00',  # 橙色
                                       **btn_style)
        self.upload_button.pack(anchor='nw', pady=(0, 10), padx=10)

        self.realtime_button = tk.Button(left_top_frame, text="实时识别",
                                         command=self.start_realtime_mode,
                                         bg='#388E3C',  # 绿色
                                         **btn_style)
        self.realtime_button.pack(anchor='nw', pady=(0, 10), padx=10)

        self.stop_button = tk.Button(left_top_frame, text="结束识别",
                                     command=self.stop_current_mode,
                                     bg='#D32F2F',  # 红色
                                     **btn_style)
        self.stop_button.pack(anchor='nw', pady=(0, 10), padx=10)

        # 分析结果信息框
        result_frame = tk.LabelFrame(left_top_frame, text="分析结果", bd=2, fg='#333',
                                     font=("Microsoft YaHei", 12, 'bold'),
                                     bg='#FFFDE7', labelanchor='n')
        result_frame.pack(fill='x', padx=10, pady=(5, 10))

        self.result_label = tk.Label(result_frame,
                                     text="识别结果：暂无\nFACS编码分析：暂无\n情绪空间位置：暂无",
                                     font=("Microsoft YaHei", 10),
                                     bg='#FFFDE7', fg="#444", justify='left')
        self.result_label.pack(anchor='nw', padx=5, pady=5)

        # ========== 左中(emoji,横向) ==========
        self.emoji_frame = tk.Frame(left_frame, bg='#FFF8E1', bd=1, relief='sunken', height=100)
        self.emoji_frame.grid(row=1, column=0, sticky='nsew', pady=5, padx=5)
        self.emoji_frame.grid_propagate(False)

        # 横向容器
        emoji_container = tk.Frame(self.emoji_frame, bg='#FFF8E1')
        emoji_container.pack(fill='both', expand=True)

        tk.Label(emoji_container, text="emoji形象:", font=("Microsoft YaHei", 12, 'bold'),
                 bg='#FFF8E1').pack(side='left', padx=5, pady=5)

        self.emoji_canvas = tk.Canvas(emoji_container, bg='#FFFFFF', highlightthickness=1, width=80, height=80)
        self.emoji_canvas.pack(side='left', padx=5, pady=5)

        # ========== 左中(语音文本滚动显示) ==========
        # 和 emoji_frame 同一位置，但默认隐藏，供“实时模式”使用
        self.text_speech_frame = tk.Frame(left_frame, bg='#FFF8E1', bd=1, relief='sunken', height=100)
        self.text_speech_frame.grid(row=1, column=0, sticky='nsew', pady=5, padx=5)
        self.text_speech_frame.grid_propagate(False)

        # 标题
        self.text_speech_label = tk.Label(
            self.text_speech_frame,
            text="实时语音文本",
            font=("Microsoft YaHei", 12, 'bold'),
            bg='#FFF8E1'
        )
        self.text_speech_label.pack(anchor='nw', padx=5, pady=5)

        # 再创建一个容器来放滚动条和文本框
        scroll_container = tk.Frame(self.text_speech_frame, bg='#FFF8E1')
        scroll_container.pack(expand=True, fill='both')

        scrollbar = tk.Scrollbar(scroll_container)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.speech_text_box = tk.Text(scroll_container, wrap='word', height=3, yscrollcommand=scrollbar.set)
        self.speech_text_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar.config(command=self.speech_text_box.yview)

        # 默认先隐藏 text_speech_frame，保留 emoji_frame
        self.text_speech_frame.grid_remove()

        self.speech_queue = queue.Queue()

        def poll_speech_queue():
            try:
                while True:
                    text = self.speech_queue.get_nowait()
                    self.speech_text_box.insert(tk.END, text + "\n")
                    self.speech_text_box.see(tk.END)
            except queue.Empty:
                pass
            self.root.after(50, poll_speech_queue)

        self.root.after(50, poll_speech_queue)

        # ========== 左下(图片/实时视频) ==========
        left_bottom_frame = tk.Frame(left_frame, bg='#FFF8E1', bd=1, relief='sunken')
        left_bottom_frame.grid(row=2, column=0, sticky='nsew', pady=(0, 5), padx=5)

        # 创建水平和垂直滚动条
        self.h_scrollbar = tk.Scrollbar(left_bottom_frame, orient='horizontal')
        self.v_scrollbar = tk.Scrollbar(left_bottom_frame, orient='vertical')
        self.h_scrollbar.pack(side='bottom', fill='x')
        self.v_scrollbar.pack(side='right', fill='y')

        # 创建Canvas并绑定两个滚动条
        self.image_canvas = tk.Canvas(
            left_bottom_frame,
            bg='#FFFFFF',
            highlightthickness=1,
            xscrollcommand=self.h_scrollbar.set,
            yscrollcommand=self.v_scrollbar.set
        )
        self.image_canvas.pack(side='left', expand=True, fill='both', padx=5, pady=5)
        self.h_scrollbar.config(command=self.image_canvas.xview)
        self.v_scrollbar.config(command=self.image_canvas.yview)

        # 创建内部Frame并绑定到Canvas
        self.inner_canvas_frame = tk.Frame(self.image_canvas, bg='#FFFFFF')
        self.canvas_frame_window = self.image_canvas.create_window(
            (0, 0), window=self.inner_canvas_frame, anchor='nw'
        )

        # 设置scrollregion绑定事件
        def update_scrollregion(event):
            self.image_canvas.configure(scrollregion=self.image_canvas.bbox("all"))

        self.inner_canvas_frame.bind("<Configure>", update_scrollregion)

        # ========== 右上(雷达图, 标题随模式) ==========
        self.radar_frame = tk.LabelFrame(right_frame, text="情绪轮分析", bd=2, relief='ridge',
                                         font=("Microsoft YaHei", 12, 'bold'), fg='#333', bg='#F1F8E9')
        self.radar_frame.grid(row=0, column=0, sticky='nsew', pady=(0, 5), padx=5)

        self.radar_canvas = tk.Canvas(self.radar_frame, bg='#FFFFFF', highlightthickness=0)
        self.radar_canvas.pack(expand=True, fill='both')

        # ========== 右中(直方图, 标题随模式) ==========
        self.prob_frame = tk.LabelFrame(right_frame, text="表情概率分布", bd=2, relief='ridge',
                                        font=("Microsoft YaHei", 12, 'bold'), fg='#333', bg='#F1F8E9')
        self.prob_frame.grid(row=1, column=0, sticky='nsew', pady=5, padx=5)

        self.prob_canvas = tk.Canvas(self.prob_frame, bg='#FFFFFF', highlightthickness=0)
        self.prob_canvas.pack(expand=True, fill='both')

        # ========== 右下(第三方框, va_canvas or time_log, 标题随模式) ==========
        self.va_frame = tk.LabelFrame(right_frame, text="效价-唤醒度分析", bd=2, relief='ridge',
                                      font=("Microsoft YaHei", 12, 'bold'), fg='#333', bg='#F1F8E9')
        self.va_frame.grid(row=2, column=0, sticky='nsew', pady=(0, 5), padx=5)

        self.va_canvas = None
        self.time_log = None
        self.canvas_face = None
        self.canvas_prob = None
        self.canvas_va = None
        self.speech_client = None
        self.speech_thread = None

        # 初始化人脸检测器
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

        # 创建 line_chart, bar_chart
        self.line_chart = EmotionLineChart(self.radar_canvas, class_names, max_duration=300)
        self.bar_chart = EmotionBarChart(self.prob_canvas, class_names, max_duration=300)
        btn_style = dict(font=("Microsoft YaHei", 12, 'bold'), fg='white', width=15, relief='raised')

        self.save_button = tk.Button(left_top_frame, text="保存数据", command=self.save_data, bg='#1976D2', state='disabled', **btn_style)
        # self.save_button.pack(anchor='nw', pady=(0,10), padx=10)
        self.save_button.pack(anchor='nw', side='left', pady=(0, 10), padx=(10, 5))

        self.analyze_button = tk.Button(
            left_top_frame, text="数据分析",
            command=self.analyze_text_emotions,
            bg='#7B1FA2',  # 紫色
            state='disabled', **btn_style)
        # self.analyze_button.pack(anchor='nw', pady=(0, 10), padx=10)
        self.analyze_button.pack(anchor='nw', side='left', pady=(0, 10), padx=(0, 10))

        # RealTimeRecognizer实例化时传入record_callback
        self.rt_recognizer = RealTimeRecognizer(
            self.image_canvas,
            model_path='FER2013_VGG19/PrivateTest_model.t7',
            line_chart=self.line_chart,
            bar_chart=self.bar_chart,
            record_callback=lambda timestamp, predicted_class, probabilities: [
                self.data_recorder.log_emotion(timestamp, predicted_class, probabilities),
                self.emotion_summary.record_emotion(predicted_class)
            ]
        )

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ================= 模式切换 =================

    def start_image_mode(self):
        """单张图片模式, 改右侧标题为: 情绪轮分析 / 表情概率分布 / 效价-唤醒度分析"""
        if self.current_mode == 'realtime':
            self.rt_recognizer.stop()
            self.save_button.config(state='disabled')
        self.clear_all()
        self.current_mode = 'image'

        # 显示 emoji_frame, 隐藏 text_speech_frame
        self.emoji_frame.grid(row=1, column=0, sticky='nsew', pady=5, padx=5)
        self.text_speech_frame.grid_remove()

        # 修改右侧标题
        self.radar_frame.config(text="情绪轮分析")
        self.prob_frame.config(text="表情概率分布")
        self.va_frame.config(text="效价-唤醒度分析")

        # 创建va_canvas
        self.va_canvas = tk.Canvas(self.va_frame, bg='#FFFFFF', highlightthickness=1)
        self.va_canvas.pack(expand=True, fill='both', padx=5, pady=5)

        # 修改标题并加载图片识别功能（同之前，无任何省略）
        file_path = filedialog.askopenfilename(title="选择图片", filetypes=[("图片文件", "*.jpg *.png *.jpeg")])
        if file_path:
            self.show_result(file_path)

    # def on_speech_text(self, recognized_text):
    #     """
    #     当语音识别到新文本时，这个方法会被调用。
    #     我们把 recognized_text 追加到左中区域的滚动文本框中。
    #     """
    #     # 将文本追加到文本框末尾，并换行
    #     self.speech_text_box.insert(tk.END, recognized_text + "\n")
    #     # 让文本框自动滚动到最新内容
    #     self.speech_text_box.see(tk.END)

    def on_speech_text(self, recognized_text):
        if recognized_text:
            self.speech_queue.put(recognized_text)

    def start_realtime_mode(self):
        """实时识别模式, 改右侧标题为: 5分钟内情绪变化趋势 / 表情概率分布 / 5分钟内情绪时间轴滚动日志"""
        if self.current_mode == 'image':
            self.clear_all()

        if self.current_mode == 'realtime':
            messagebox.showinfo("提示", "已在实时模式中")
            return

        self.clear_all()
        self.current_mode = 'realtime'

        # 隐藏 emoji_frame, 显示 text_speech_frame
        self.emoji_frame.grid_remove()
        self.text_speech_frame.grid(row=1, column=0, sticky='nsew', pady=5, padx=5)

        # 修改标题为实时识别模式
        self.radar_frame.config(text="5分钟内情绪变化趋势")
        self.prob_frame.config(text="5分钟内情绪占比")
        self.va_frame.config(text="5分钟内情绪时间轴滚动日志")

        self.time_log = TimeLog(self.va_frame, max_duration=300)
        self.rt_recognizer.time_log = self.time_log

        self.save_button.config(state='normal')

        try:
            self.rt_recognizer.start()
            # 1) 包一层调试回调：先在控制台打印，再交给 on_speech_text
            def debug_cb(t):
                # print("ASR_CALLBACK:", repr(t))
                self.on_speech_text(t)

            # 2) 用 debug_cb 替代原来的 self.on_speech_text
            self.speech_client = IatWsClient(
                appid=os.getenv('XFYUN_APPID', ''),
                api_key=os.getenv('XFYUN_API_KEY', ''),
                api_secret=os.getenv('XFYUN_API_SECRET', ''),
                text_callback=debug_cb
            )

            # 3) 线程启动保持不变
            self.speech_thread = threading.Thread(target=self.speech_client.start, daemon=True)
            self.speech_thread.start()

            self.result_label.config(text="实时人脸和语音识别已开始")
        except RuntimeError as e:
            messagebox.showerror("错误", str(e))
            self.current_mode = 'none'

    def stop_current_mode(self):
        """结束识别"""
        if self.current_mode == 'image':
            self.clear_all()
            self.current_mode = 'none'
            self.result_label.config(text="图片识别已结束")
        elif self.current_mode == 'realtime':
            self.rt_recognizer.stop()
            # 停止语音转文字模块
            if self.speech_client:
                self.speech_client.stop()
            if self.speech_thread:
                self.speech_thread.join(timeout=2)
            self.speech_client = None
            self.speech_thread = None

            self.save_button.config(state='disabled')
            # 此处生成情绪总结
            summary_text = self.emotion_summary.summarize_emotions()
            self.result_label.config(text=summary_text)

            # 生成后重置情绪记录，以便下次识别
            self.emotion_summary.reset_summary()
            self.clear_all()
            self.current_mode = 'none'
            self.result_label.config(text="实时识别已结束")
        else:
            messagebox.showinfo("提示", "当前没有正在进行的识别")

    # ================= 保存数据 =================

    def save_data(self):
        # 保存人脸识别数据
        self.data_recorder.save_data()

        # 如果语音转文字模块存在，则保存识别到的文字
        if self.speech_client:
            self.speech_client.save_transcription()

        # 实时情绪总结并更新结果展示
        emotion_summary_text = self.emotion_summary.summarize_emotions()
        self.result_label.config(text=emotion_summary_text)

        # 弹出提示
        messagebox.showinfo("提示", "数据保存成功，已停止实时识别。")

        # 停止实时识别，但不清空所有内容（重要修改！）
        if self.current_mode == 'realtime':
            self.rt_recognizer.stop()

            # 停止语音识别模块
            if self.speech_client:
                self.speech_client.stop()
            if self.speech_thread:
                self.speech_thread.join(timeout=2)
            self.speech_client = None
            self.speech_thread = None

            # 更新当前模式状态
            self.current_mode = 'none'

            # 此处不再调用clear_all(), 而是选择性保留右侧三个方框内容！

            # 启用数据分析按钮（以保证数据分析正常进行）
            self.analyze_button.config(state='normal')

            # 重置情绪总结记录，以便下次识别
            self.emotion_summary.reset_summary()

        # 注意：此处不调用 clear_all()，右侧数据将保持显示

    # ================= 语音文本情绪分析 =================

    def analyze_text_emotions(self):
        base_path = os.path.join('Data Recording', 'translation-data')

        # 获取所有子文件夹，并根据创建时间排序找到最新的文件夹
        all_folders = [f.path for f in os.scandir(base_path) if f.is_dir()]
        if not all_folders:
            messagebox.showerror("错误", "未找到任何数据文件夹")
            return

        latest_folder = max(all_folders, key=os.path.getctime)

        # 最新的translation.txt完整路径
        translation_path = os.path.join(latest_folder, 'translation.txt')

        if not os.path.exists(translation_path):
            messagebox.showerror("错误", f"未找到文件: {translation_path}")
            return

        try:
            sentences, emotions, emotion_labels = self.text_emotion_analyzer.analyze_text_file(translation_path)
        except FileNotFoundError as e:
            messagebox.showerror("错误", str(e))
            return

        # 更新左中区域展示
        self.speech_text_box.delete('1.0', tk.END)
        self.text_speech_label.config(text="语音文本情绪识别展示")

        for idx, (sentence, label) in enumerate(zip(sentences, emotion_labels), 1):
            self.speech_text_box.insert(tk.END, f"{idx}. {sentence} → 【{label}】\n")
            self.speech_text_box.see(tk.END)

        # 绘制情绪变化折线图，并保存到最新文件夹内
        self.display_text_emotion_chart(emotions, emotion_labels, latest_folder)

        # 获取当前实时识别得到的人脸情绪
        face_emotions = self.data_recorder.emotion_log  # 从 log 中提取情绪标签
        face_labels = [entry["emotion"] for entry in face_emotions if "emotion" in entry]

        # 获取人脸情绪分布
        face_distribution = summarize_face_emotion_distribution(face_labels)

        # 获取文本情绪分布
        text_distribution = summarize_text_emotion_distribution(emotion_labels)

        # 融合判断
        fused_emotion = fuse_emotions(face_distribution, text_distribution)

        # 更新展示
        summary_text = fusion_summary(face_distribution, text_distribution, fused_emotion)
        self.result_label.config(text=summary_text)

    # ================= 语音文本情绪直方图绘制 =================

    def display_text_emotion_chart(self, emotions, emotion_labels, save_folder):
        plt.figure(figsize=(8, 4), dpi=100)
        plt.plot(range(1, len(emotions) + 1), emotions, marker='o', linestyle='-', color='purple')

        plt.yticks([0, 1, 2, 3, 4], ['悲伤', '快乐', '愤怒', '恐惧', '厌恶'])
        plt.xlabel('语句序号', fontsize=12)
        plt.ylabel('情绪类别', fontsize=12)
        plt.title('文本情绪变化趋势', fontsize=14)
        plt.grid(alpha=0.4)

        chart_path = os.path.join(save_folder, 'text_emotion_analysis.png')
        plt.savefig(chart_path)

        # 在左下区域展示该图
        self.image_canvas.delete('all')

        img = tk.PhotoImage(file=chart_path)
        self.image_canvas.create_image(0, 0, anchor='nw', image=img)
        self.image_canvas.image = img

        label = tk.Label(self.inner_canvas_frame, image=img, bg='#FFFFFF')
        label.pack(anchor='nw')

        # 关键点：将图片对象保存到类成员变量
        self.inner_canvas_frame.image = img  # 防止图像被自动回收

        # 主动刷新scrollregion确保滚动条有效
        self.image_canvas.update_idletasks()
        self.image_canvas.configure(scrollregion=self.image_canvas.bbox("all"))

    # ================= 清理界面 =================

    def clear_all(self):
        """
        清空左侧画布, 右侧图表, 第三个方框(va_canvas/time_log),
        以及FigureCanvasTkAgg等
        """
        self.image_canvas.delete('all')
        self.emoji_canvas.delete('all')
        # 清空文本框
        self.speech_text_box.delete('1.0', tk.END)
        for w in self.radar_canvas.winfo_children():
            w.destroy()
        for w in self.prob_canvas.winfo_children():
            w.destroy()

        if self.va_canvas:
            self.va_canvas.destroy()
            self.va_canvas = None
        if self.time_log:
            self.time_log.frame.destroy()
            self.time_log = None
            self.rt_recognizer.time_log = None

        if self.canvas_face:
            self.canvas_face.get_tk_widget().destroy()
            self.canvas_face = None
        if self.canvas_prob:
            self.canvas_prob.get_tk_widget().destroy()
            self.canvas_prob = None
        if self.canvas_va:
            self.canvas_va.get_tk_widget().destroy()
            self.canvas_va = None
        # 重置结果
        self.result_label.config(text="识别结果：暂无\nFACS编码分析：暂无\n情绪空间位置：暂无")
        self.analyze_button.config(state='disabled')
        self.text_speech_label.config(text="实时语音文本")

    # ================= 单张图片识别 =================

    def show_result(self, file_path):
        """
        对上传图片做识别, 更新左下图像/emoji, 右侧图表.
        仅在 current_mode=='image' 时使用.
        """
        predicted_class, scores, raw_img = predict_expression(file_path)
        gray_img = cv2.cvtColor(raw_img, cv2.COLOR_RGB2GRAY)
        faces = self.detector(gray_img)

        if not faces:
            messagebox.showwarning("提示", "未检测到人脸,请更换图片")
            return

        face = faces[0]
        shape = self.predictor(gray_img, face)
        landmarks = [(shape.part(i).x, shape.part(i).y) for i in range(68)]
        facs_codes = get_facs_from_landmarks(landmarks)
        facs_desc = '，'.join([f"{code}（{facs_dict[code]}）" for code in facs_codes])

        va_dict = {
            'Angry': (-0.8, 0.7), 'Disgust': (-0.6, 0.3), 'Fear': (-0.8, 0.8),
            'Happy': (0.9, 0.7), 'Sad': (-0.7, -0.5), 'Surprise': (0.2, 0.9),
            'Neutral': (0.0, 0.0)
        }
        valence_weighted = sum(scores[i] * va_dict[class_names[i]][0] for i in range(len(class_names)))
        arousal_weighted = sum(scores[i] * va_dict[class_names[i]][1] for i in range(len(class_names)))

        # 更新结果Label
        self.result_label.config(
            text=f"识别结果：{predicted_class} ({scores[class_names.index(predicted_class)] * 100:.2f}%)\n\n"
                 f"FACS编码分析：{facs_desc}\n\n"
                 f"情绪空间位置：效价 {valence_weighted:.2f}, 唤醒度 {arousal_weighted:.2f}"
        )

        # 雷达图
        emotion_map = {
            'Happy': '快乐', 'Angry': '愤怒', 'Sad': '悲伤',
            'Fear': '恐惧', 'Disgust': '厌恶', 'Surprise': '惊讶',
            'Neutral': '中性'
        }
        emotion_categories = list(emotion_map.values())
        emotion_values = [0] * len(emotion_categories)
        for i, cls_ in enumerate(class_names):
            if cls_ in emotion_map:
                idx = emotion_categories.index(emotion_map[cls_])
                emotion_values[idx] = scores[i] * 100
        draw_radar(self.radar_canvas, emotion_categories, emotion_values)

        # 左下图像 + 关键点
        self.canvas_face = draw_face(self.image_canvas, raw_img, landmarks)

        # Emoji
        draw_emoji(self.emoji_canvas, predicted_class)

        # 概率分布
        draw_probability(self.prob_canvas, class_names, scores, color_list)

        # 效价-唤醒度散点图 => va_canvas
        draw_va(self.va_canvas, va_dict, scores, class_names, color_list, valence_weighted, arousal_weighted)

    def on_close(self):
        """关闭窗口时,若在实时模式则先stop,再destroy"""
        if self.current_mode == 'realtime':
            self.rt_recognizer.stop()
        self.root.quit()
        self.root.destroy()

if __name__ == '__main__':
    root = tk.Tk()
    app = EmotionRecognitionApp(root)
    root.mainloop()
