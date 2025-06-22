# SUDAH DI UJI SECARA TERMINAL DAN BERJALAN NORMAL DAN BAGUS VIA CONSOLE

# langkah 1 :
# python -m venv venv
# source venv/bin/activate

# langkah 2 :
# pip install webrtcvad pyaudio numpy nltk rank_bm25 sentence-transformers speechrecognition gtts playsound

# langkah 3 :
# Pastikan file data/pertanyaan_jawaban.json sudah ada dan sesuai format yang diharapkan.

# langkah 4 :
# jalankan dari terminal code : python clinicbot_durasi_STT-TTS_BM25-BERT.py (pastikan dalam folter project/clinicbot ya)

# file requirement.txt sbb : 
# webrtcvad
# pyaudio
# numpy
# nltk
# rank_bm25
# sentence-transformers
# speechrecognition
# gtts
# playsound
# pydub
# pydub
# sounddevice



# langkah 5 install requierement : 
# pip install -r requirements.txt

#dan pastikan ffmpeg sudah terinstall di sistem Anda (brew install ffmpeg untuk Mac, atau via package manager lain).

# Integrasi ke Class
# Letakkan fungsi animate_waveform dan modifikasi text_to_speech sebagai method di dalam class ClinicBotGUI.
# Panggil self.text_to_speech(answer) pada bagian chatbot bicara.
# Tambahkan self.canvas pada GUI layout.


# Konsep Visualisasi KITT-Style
# Bar-graph bergerak: Bar LED (rectangle) dengan tinggi dinamis.
# Warna merah (atau gradien merah-kuning).
# Animasi bergerak dari kiri ke kanan dan kembali ("ping-pong") atau center-out.
# Sinkron dengan suara yang dikeluarkan.

# Cara Implementasi di Tkinter
# Ekstrak data amplitudo audio (pakai pydub/numpy).
# Bagi data menjadi beberapa band/bar (misal 20 bar).
# Untuk masing-masing bar, hitung energi/peak di frekuensi/bagian tertentu.
# Animasi bar: Ubah tinggi (dan/atau warna) sesuai energi band, update tiap frame (30-60 fps).
# Buat animasi “ping-pong” jika ingin efek KITT berjalan bolak-balik.


import webrtcvad
import pyaudio
import wave
import time
import numpy as np
import nltk
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util
import json
import speech_recognition as sr
from gtts import gTTS
import os
import threading
import tkinter as tk

try:
    import sounddevice as sd
    USE_SD = True
except ImportError:
    USE_SD = False

def play_beep(frequency=600, duration=0.2, samplerate=44100):
    if not USE_SD:
        try:
            if os.name == 'nt':
                import winsound
                winsound.Beep(int(frequency), int(duration*1000))
            else:
                os.system('printf "\a"')
        except Exception:
            pass
        return
    t = np.linspace(0, duration, int(samplerate*duration), False)
    tone = np.sin(frequency * t * 2 * np.pi)
    audio = tone * (2**15 - 1) / np.max(np.abs(tone))
    audio = audio.astype(np.int16)
    sd.play(audio, samplerate)
    sd.wait()

with open("data/q&a.json", "r") as f:
    pairs = json.load(f)

pertanyaan_list = [item["pertanyaan"] for item in pairs]
jawaban_list = [item["jawaban"] for item in pairs]

nltk.download('punkt')
def tokenize(text):
    return nltk.word_tokenize(text.lower())

corpus_tokenized = [tokenize(q) for q in pertanyaan_list]
bm25 = BM25Okapi(corpus_tokenized)
sbert_model = SentenceTransformer('distiluse-base-multilingual-cased-v2')
corpus_embeddings = sbert_model.encode(pertanyaan_list, convert_to_tensor=True)

def get_bm25_answer(query):
    scores = bm25.get_scores(tokenize(query))
    best_idx = int(np.argmax(scores))
    return jawaban_list[best_idx]

def get_sbert_answer(query):
    query_embedding = sbert_model.encode(query, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
    best_idx = int(similarities.argmax())
    return jawaban_list[best_idx]

def get_hybrid_answer(query):
    query_embedding = sbert_model.encode(query, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
    if similarities.max().item() > 0.55:
        return get_sbert_answer(query)
    else:
        return get_bm25_answer(query)

VAD_LEVEL = 2
SILENCE_TIMEOUT = 1.2
CHANNELS = 1
RATE = 16000
FRAME_DURATION = 30
FRAME_SIZE = int(RATE * FRAME_DURATION / 1000)
FORMAT = pyaudio.paInt16

vad = webrtcvad.Vad(VAD_LEVEL)

def record_once_vad(pa, stream):
    audio_frames = []
    recording = False
    last_voice_time = None
    start_time = time.time()
    while True:
        frame = stream.read(FRAME_SIZE, exception_on_overflow=False)
        is_speech = vad.is_speech(frame, RATE)
        now = time.time()
        if is_speech:
            if not recording:
                recording = True
            audio_frames.append(frame)
            last_voice_time = now
        elif recording:
            audio_frames.append(frame)
            if last_voice_time and (now - last_voice_time > SILENCE_TIMEOUT):
                break
        if (not recording) and (now - start_time > 8):
            break
    if audio_frames:
        filename = "output.wav"
        wf = wave.open(filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(pa.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(audio_frames))
        wf.close()
        return filename
    else:
        return None

def wav_to_text_google(filename):
    r = sr.Recognizer()
    with sr.AudioFile(filename) as source:
        audio = r.record(source)
    try:
        text = r.recognize_google(audio, language="id-ID")
        return text
    except sr.UnknownValueError:
        return None
    except sr.RequestError as e:
        return None

from pydub import AudioSegment

class ClinicBotGUI:
    def __init__(self, master):
        self.master = master
        master.title("KlinikBot Voice Chat (GUI)")
        self.label = tk.Label(master, text="Klik 'Mulai Mic' agar mic standby otomatis. Klik 'Stop Mic' untuk berhenti.")
        self.label.pack(pady=10)
        self.start_button = tk.Button(master, text="Mulai Mic", command=self.start_mic)
        self.start_button.pack(pady=5)
        self.stop_button = tk.Button(master, text="Stop Mic", command=self.stop_mic, state="disabled")
        self.stop_button.pack(pady=5)
        self.textbox = tk.Text(master, height=10, width=70, state="disabled")
        self.textbox.pack(pady=10)
        # Tinggi canvas lebih besar agar dua arah tampak jelas
        self.canvas = tk.Canvas(master, width=350, height=500, bg="black")
        self.canvas.pack(pady=5)
        self.running = False
        self.mic_thread = None

    def start_mic(self):
        if self.running:
            return
        self.running = True
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.append_text("Mic standby... Silakan bicara kapan saja.")
        self.mic_thread = threading.Thread(target=self.voice_loop)
        self.mic_thread.daemon = True
        self.mic_thread.start()

    def stop_mic(self):
        self.running = False
        self.append_text("Mic dimatikan.")
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")

    def voice_loop(self):
        pa = pyaudio.PyAudio()
        stream = pa.open(format=FORMAT,
                         channels=CHANNELS,
                         rate=RATE,
                         input=True,
                         frames_per_buffer=FRAME_SIZE)
        try:
            while self.running:
                wavfile = record_once_vad(pa, stream)
                if not self.running:
                    break
                if not wavfile:
                    continue
                query = wav_to_text_google(wavfile)
                os.remove(wavfile)
                if not query:
                    err = "Tolong diulangi lagi, saya tidak mendengar secara jelas."
                    self.append_text(f"[Chatbot] {err}")
                    self.text_to_speech(err)
                    continue
                self.append_text(f"[Anda] {query}")
                answer = get_hybrid_answer(query)
                self.append_text(f"[Chatbot] {answer}")
                self.text_to_speech(answer)
                self.append_text("="*40)
        finally:
            stream.stop_stream()
            stream.close()
            pa.terminate()
            self.stop_mic()

    # Center-out/inside-out bar spectrum (Knight Rider style) - DUA ARAH ATAS BAWAH
    def animate_centerout_spectrum_dual(self, filename, stop_event, n_bars=15, n_levels=50, mode="center-out"):
        try:
            audio = AudioSegment.from_file(filename)
            samples = np.array(audio.get_array_of_samples())
            if audio.channels == 2:
                samples = samples[::2]
            samples = samples.astype(np.float32)
            samples /= np.max(np.abs(samples)) + 1e-6
            length = len(samples)
            duration = audio.duration_seconds
            frames = int(duration * 20)
            samples_per_frame = max(1, length // frames)
            prev_val = 0
            w = int(self.canvas['width'])
            h = int(self.canvas['height'])
            bar_width = int(w // (n_bars + 2))
            bar_height = int((h * 6) // 3)  # Setengah tinggi canvas
            level_height = max(1, bar_height // n_levels)
            center_idx = n_bars // 2
            center_y = h // 2

            for i in range(frames):
                if stop_event.is_set():
                    break
                self.canvas.delete("wave")
                start = i * samples_per_frame
                end = start + samples_per_frame
                chunk = samples[start:end]
                if len(chunk) < 2:
                    continue
                val = np.sqrt(np.mean(chunk ** 2))
                val = 0.6 * prev_val + 0.4 * val
                prev_val = val
                val = min(val, 1.0)
                n_lit = int(val * n_levels)
                bar_vals = [0] * n_bars

                if mode == "center-out":
                    for idx in range(center_idx, n_bars):
                        bar_vals[idx] = max(n_lit - (idx - center_idx) * 2, 0)
                    for idx in range(center_idx-1, -1, -1):
                        bar_vals[idx] = max(n_lit - (center_idx - idx) * 2, 0)
                elif mode == "outside-in":
                    for idx in range(center_idx+1):
                        bar_vals[idx] = max(n_lit - idx * 2, 0)
                    for idx in range(center_idx+1, n_bars):
                        bar_vals[idx] = max(n_lit - (n_bars - 1 - idx) * 2, 0)

                for idx in range(n_bars):
                    x0 = (w//2 - bar_width//2) + (idx - center_idx) * (bar_width + 4)
                    # Atas dari tengah ke atas
                    for lvl in range(bar_vals[idx]):
                        y0 = center_y - lvl * level_height
                        y1 = y0 - level_height + 2
                        color = "#FFC107"
                        self.canvas.create_rectangle(
                            x0, y1, x0 + bar_width, y0,
                            fill=color, width=0, tags="wave"
                        )
                    # Bawah dari tengah ke bawah
                    for lvl in range(bar_vals[idx]):
                        y0 = center_y + lvl * level_height
                        y1 = y0 + level_height - 2
                        color = "#e53935"
                        self.canvas.create_rectangle(
                            x0, y0, x0 + bar_width, y1,
                            fill=color, width=0, tags="wave"
                        )
                self.canvas.update()
                self.canvas.after(int(1000/30))
            self.canvas.delete("wave")
        except Exception as e:
            print("Center-out dual spectrum error:", e)
            self.canvas.delete("wave")

    def text_to_speech(self, text, lang="id"):
        play_beep()
        tts = gTTS(text, lang=lang)
        filename = "tts_output.mp3"
        tts.save(filename)
        stop_event = threading.Event()
        spectrum_thread = threading.Thread(
            target=self.animate_centerout_spectrum_dual,
            args=(filename, stop_event),
            kwargs={"mode": "center-out"} # "outside-in" juga bisa
        )
        spectrum_thread.start()
        try:
            from playsound import playsound
            playsound(filename)
        except Exception:
            os.system(f"open {filename}")
        stop_event.set()
        spectrum_thread.join()
        os.remove(filename)

    def append_text(self, msg):
        self.textbox.config(state="normal")
        self.textbox.insert(tk.END, msg + "\n")
        self.textbox.see(tk.END)
        self.textbox.config(state="disabled")

if __name__ == "__main__":
    root = tk.Tk()
    app = ClinicBotGUI(root)
    root.mainloop()