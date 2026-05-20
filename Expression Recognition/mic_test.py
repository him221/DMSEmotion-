import sounddevice as sd
import numpy as np
import time

sd.default.samplerate = 16000
sd.default.channels = 1

print("Devices:")
print(sd.query_devices())
print("Default input device:", sd.default.device)

print("Recording 3 seconds... 请对着麦克风说话")
audio = sd.rec(int(3 * 16000), dtype="float32")
sd.wait()

rms = float(np.sqrt(np.mean(audio**2)))
mx = float(np.max(np.abs(audio)))
print("RMS:", rms, "MAX:", mx)

if mx < 0.01:
    print("几乎是静音：可能没录到声音/设备选错/权限/独占模式/采样率不匹配")
else:
    print("麦克风采集正常")