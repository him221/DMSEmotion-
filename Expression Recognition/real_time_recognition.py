import cv2
import dlib
import torch
import torch.nn.functional as F
import transforms as transforms
import numpy as np
from PIL import Image, ImageTk
from models import VGG
from image_recognition import class_names
import datetime

class RealTimeRecognizer:
    def __init__(self, canvas, model_path='FER2013_VGG19/PrivateTest_model.t7', line_chart=None, bar_chart=None, time_log=None, record_callback=None):
        self.canvas = canvas
        self.cap = None
        self.is_running = False
        self.line_chart = line_chart
        self.bar_chart = bar_chart
        self.time_log = time_log
        self.record_callback = record_callback

        self.detector = dlib.get_frontal_face_detector()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = VGG('VGG19').to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.net.load_state_dict(checkpoint['net'])
        self.net.eval()

        self.tkimg = None
        self.frame_count = 0

    def start(self):
        if self.is_running:
            return
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError("无法打开摄像头")
        self.is_running = True
        self.update_frame()

    def stop(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.canvas.delete('all')

    def update_frame(self):
        if not self.is_running or self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.canvas.after(50, self.update_frame)
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = self.detector(frame_rgb, 0)
        predicted_class = "Neutral"

        for face in faces:
            cv2.rectangle(frame_rgb, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
            predicted_class, scores = self.predict_emotion(frame_rgb, face)

            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            probabilities = dict(zip(class_names, scores))

            if self.record_callback:
                self.record_callback(timestamp, predicted_class, probabilities)

            cv2.putText(frame_rgb, f"Emotion: {predicted_class}", (face.left(), max(face.top()-10,20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        if self.line_chart:
            self.line_chart.add_emotion(predicted_class)

        if self.bar_chart:
            self.bar_chart.add_emotion(predicted_class)

        if self.time_log:
            self.time_log.add_emotion(predicted_class)

        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        h, w, _ = frame_rgb.shape
        aspect_frame = w / h
        aspect_canvas = canvas_width / canvas_height

        if aspect_canvas > aspect_frame:
            new_height = canvas_height
            new_width = int(new_height * aspect_frame)
        else:
            new_width = canvas_width
            new_height = int(new_width / aspect_frame)

        frame_resized = cv2.resize(frame_rgb, (new_width, new_height), interpolation=cv2.INTER_AREA)
        pil_img = Image.fromarray(frame_resized)
        self.tkimg = ImageTk.PhotoImage(image=pil_img)

        self.canvas.delete('all')
        offset_x = (canvas_width - new_width) // 2
        offset_y = (canvas_height - new_height) // 2
        self.canvas.create_image(offset_x, offset_y, image=self.tkimg, anchor='nw')

        self.frame_count += 1
        if self.frame_count % 10 == 0:
            if self.line_chart:
                self.line_chart.update_chart()
            if self.bar_chart:
                self.bar_chart.update_chart()
            if self.time_log:
                self.time_log.update_log()

        self.canvas.after(50, self.update_frame)

    def stop(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.canvas.delete('all')

    def predict_emotion(self, frame_rgb, face):
        x1, y1 = max(0, face.left()), max(0, face.top())
        x2, y2 = min(frame_rgb.shape[1], face.right()), min(frame_rgb.shape[0], face.bottom())
        face_roi = frame_rgb[y1:y2, x1:x2]
        face_gray = cv2.cvtColor(face_roi, cv2.COLOR_RGB2GRAY)
        face_gray = cv2.resize(face_gray, (48, 48), interpolation=cv2.INTER_AREA)
        face_3ch = np.stack([face_gray]*3, axis=-1)
        pil_face = Image.fromarray(face_3ch)
        transform_test = transforms.Compose([
            transforms.TenCrop(44),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]))
        ])
        inputs = transform_test(pil_face).to(self.device)
        with torch.no_grad():
            outputs = self.net(inputs)
        outputs_avg = outputs.view(10, -1).mean(0)
        scores = F.softmax(outputs_avg, dim=0).cpu().numpy()
        predicted_idx = np.argmax(scores)
        return class_names[predicted_idx], scores
