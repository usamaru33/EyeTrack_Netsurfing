import pyaudio
import numpy as np
import pyautogui

# 音声キャプチャの設定
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
THRESHOLD = 2000  # 拍手の音量を検知するための閾値
CLAP_DURATION = 0.1  # 拍手の持続時間（秒）

def detect_clap(data):
    """音声データから拍手を検出する関数"""
    audio_data = np.frombuffer(data, dtype=np.int16)
    peak = np.max(np.abs(audio_data))

    if peak > THRESHOLD:
        return True
    return False

# PyAudioの初期化
p = pyaudio.PyAudio()

# ストリームの開始
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("拍手を検知しています...")

try:
    while True:
        data = stream.read(CHUNK)
        if detect_clap(data):
            print("拍手を検知しました！")
            pyautogui.click()
except KeyboardInterrupt:
    print("終了します...")

# ストリームの停止と解放
stream.stop_stream()
stream.close()
p.terminate()