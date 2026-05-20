import threading
from tkinter import messagebox

import websocket
import hashlib
import base64
import hmac
import json
from urllib.parse import urlencode
import time
import ssl
from wsgiref.handlers import format_date_time
from datetime import datetime
from time import mktime
import logging
import queue
import os
import numpy as np
import sounddevice as sd

STATUS_FIRST_FRAME = 0
STATUS_CONTINUE_FRAME = 1


class IatWsClient:
    """
    - 录音：sounddevice (16k, int16, mono)
    - 发送：WebSocket IAT
    - 目标：
        1) 不打印 SD_AUDIO
        2) 有弹窗提示
        3) 允许长沉默不轻易断
        4) 不打印 ASR_CALLBACK
    """

    def __init__(
        self,
        appid: str,
        api_key: str,
        api_secret: str,
        text_callback=None,
        device_index: int = 1,       # 你当前验证成功的设备
        rate: int = 16000,
        block_ms: int = 20,
        vad_eos_ms: int = 60000,     # ✅ 允许 60 秒沉默才认为一句结束
        keepalive_silence: bool = True,   # ✅ 沉默时发送静音帧保活
        silence_send_interval: float = 0.5 # 每0.5秒发一次短静音
    ):
        self.appid = appid
        self.api_key = api_key
        self.api_secret = api_secret
        self.text_callback = text_callback

        self.ws_url = self.create_url()
        self.ws = None

        self.running = False
        self._session_alive = False

        self.transcription_buffer = []
        self.audio_q: "queue.Queue[bytes]" = queue.Queue(maxsize=800)

        self.RATE = rate
        self.CHANNELS = 1
        self.BLOCK = int(self.RATE * block_ms / 1000)  # 20ms -> 320
        self.device_index = device_index

        self.vad_eos_ms = vad_eos_ms
        self.keepalive_silence = keepalive_silence
        self.silence_send_interval = silence_send_interval

        # 静音帧（20ms int16 mono）
        self._silence_frame = (np.zeros((self.BLOCK, 1), dtype=np.int16)).tobytes()

        # 防止弹窗重复刷屏
        self._last_popup_t = 0.0

    def create_url(self):
        url = "wss://ws-api.xfyun.cn/v2/iat"
        now = datetime.now()
        date = format_date_time(mktime(now.timetuple()))

        signature_origin = (
            "host: ws-api.xfyun.cn\n"
            f"date: {date}\n"
            "GET /v2/iat HTTP/1.1"
        )
        signature_sha = hmac.new(
            self.api_secret.encode("utf-8"),
            signature_origin.encode("utf-8"),
            digestmod=hashlib.sha256
        ).digest()
        signature_sha = base64.b64encode(signature_sha).decode("utf-8")

        authorization_origin = (
            f'api_key="{self.api_key}", algorithm="hmac-sha256", '
            f'headers="host date request-line", signature="{signature_sha}"'
        )
        authorization = base64.b64encode(authorization_origin.encode("utf-8")).decode("utf-8")

        params = {"authorization": authorization, "date": date, "host": "ws-api.xfyun.cn"}
        return url + "?" + urlencode(params)

    # ================= WebSocket 回调 =================

    def on_message(self, ws, message):
        try:
            msg = json.loads(message)

            if msg.get("code", 0) != 0:
                code = msg.get("code")
                sid = msg.get("sid", "")
                errMsg = msg.get("message", "")
                logging.error(f"sid:{sid} 调用出错: {errMsg}, code: {code}")

                # ✅ 10165：弹窗提示（但不要无限弹）
                if code == 10165:
                    now = time.time()
                    if now - self._last_popup_t > 3.0:
                        self._last_popup_t = now
                        try:
                            messagebox.showinfo("提示", "长时间未识别到语音，语音转文字进程停止。")
                        except Exception:
                            pass

                # 结束本轮，避免继续send导致 invalid handle
                self.running = False
                try:
                    ws.close()
                except Exception:
                    pass
                return

            ws_result = msg.get("data", {}).get("result", {}).get("ws", [])
            result = ""

            # ✅ 只取最优候选（按 sc 最大）
            for group in ws_result:
                cws = group.get("cw", [])
                if not cws:
                    continue
                best = max(cws, key=lambda x: x.get("sc", 0))
                result += best.get("w", "")

            recognized_text = result.strip()
            if recognized_text and recognized_text not in ["。", ".。", " .。", " 。"]:
                print("识别结果:", recognized_text)  # 你如果连这个也不要，我也能去掉
                self.transcription_buffer.append(recognized_text)
                if self.text_callback:
                    self.text_callback(recognized_text)

        except Exception:
            logging.exception("处理返回消息异常:")

    def on_error(self, ws, error):
        logging.error(f"WebSocket 错误: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        logging.info("WebSocket 连接关闭。")
        print("WebSocket 连接关闭。")

    def on_open(self, ws):
        self._session_alive = True
        print("WebSocket 已打开，开始录音……")

        # ========== 录音线程 ==========
        def audio_capture():
            try:
                def callback(indata, frames, time_info, status):
                    # ✅ 不打印 SD_AUDIO
                    if not self.running:
                        return
                    data_bytes = indata.tobytes()
                    try:
                        self.audio_q.put_nowait(data_bytes)
                    except queue.Full:
                        pass

                with sd.InputStream(
                    samplerate=self.RATE,
                    channels=self.CHANNELS,
                    dtype="int16",
                    blocksize=self.BLOCK,
                    device=self.device_index,
                    callback=callback
                ):
                    while self.running:
                        time.sleep(0.05)
            except Exception as e:
                logging.error(f"audio_capture exception: {e!r}")
            finally:
                # 录音退出不用打印
                pass

        # ========== 发送线程 ==========
        def send_audio():
            status = STATUS_FIRST_FRAME
            last_silence_send = 0.0

            while self.running:
                if not ws or not ws.sock or not ws.sock.connected:
                    break

                buf = None
                try:
                    # 取音频帧，最多等 0.2 秒
                    buf = self.audio_q.get(timeout=0.2)
                except queue.Empty:
                    buf = None

                # ✅ 沉默保活：没拿到音频帧时，周期性发一帧静音
                if buf is None:
                    if self.keepalive_silence:
                        now = time.time()
                        if now - last_silence_send >= self.silence_send_interval:
                            buf = self._silence_frame
                            last_silence_send = now
                        else:
                            continue
                    else:
                        continue

                try:
                    if status == STATUS_FIRST_FRAME:
                        payload = {
                            "common": {"app_id": self.appid},
                            "business": {
                                "domain": "iat",
                                "language": "zh_cn",
                                "accent": "mandarin",
                                "vinfo": 1,
                                "vad_eos": self.vad_eos_ms
                            },
                            "data": {
                                "status": 0,
                                "format": "audio/L16;rate=16000",
                                "audio": base64.b64encode(buf).decode("utf-8"),
                                "encoding": "raw"
                            }
                        }
                        ws.send(json.dumps(payload))
                        status = STATUS_CONTINUE_FRAME
                    else:
                        payload = {
                            "data": {
                                "status": 1,
                                "format": "audio/L16;rate=16000",
                                "audio": base64.b64encode(buf).decode("utf-8"),
                                "encoding": "raw"
                            }
                        }
                        ws.send(json.dumps(payload))
                except Exception as e:
                    logging.error(f"ws.send failed: {e!r}")
                    break

            # 结束线程时不做多余打印

        threading.Thread(target=audio_capture, daemon=True).start()
        threading.Thread(target=send_audio, daemon=True).start()

    # ================= 对外接口 =================

    def start(self):
        self.running = True
        self.ws = websocket.WebSocketApp(
            self.ws_url,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
            on_open=self.on_open
        )
        self.ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE}, ping_interval=30, ping_timeout=10)

    def stop(self):
        self.running = False
        # 清空队列
        try:
            while not self.audio_q.empty():
                self.audio_q.get_nowait()
        except Exception:
            pass
        if self.ws:
            try:
                self.ws.close()
            except Exception:
                pass

    def save_transcription(self):
        # 你原来的保存逻辑可以照用
        if not self.transcription_buffer:
            print("没有识别文本需要保存。")
            return

        base_folder = os.path.join("Data Recording", "translation-data")
        date_time_folder = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        final_folder_path = os.path.join(base_folder, date_time_folder)
        os.makedirs(final_folder_path, exist_ok=True)
        file_path = os.path.join(final_folder_path, "translation.txt")

        with open(file_path, "a", encoding="utf-8") as f:
            for line in self.transcription_buffer:
                f.write(line + "\n")

        print(f"已保存 {len(self.transcription_buffer)} 条识别文本到 {file_path}")
        self.transcription_buffer.clear()