import os
import math
import queue
import json
import threading
import tkinter as tk
from tkinter import scrolledtext
import pyaudio
import requests
from vosk import Model, KaldiRecognizer

class JouleVttApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Joule VTT - Voice Controller")
        self.root.geometry("500x600")
        self.root.configure(bg="#1e1e2e")

        self.is_listening = False
        self.animation_running = False
        self.anim_step = 0
        self.vosk_loaded = False
        
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        self.CHUNK = 4096
        
        self.audio_queue = queue.Queue()
        self.pyaudio_instance = pyaudio.PyAudio()
        self.server_url = "http://127.0.0.1:5000/api/voice_input"

        self.setup_ui()
        threading.Thread(target=self.load_vosk_model, daemon=True).start()

    def setup_ui(self):
        title = tk.Label(self.root, text="JOULE VOICE INPUT", font=("Helvetica", 16, "bold"), fg="#cdd6f4", bg="#1e1e2e")
        title.pack(pady=20)

        self.canvas = tk.Canvas(self.root, width=400, height=100, bg="#181825", highlightthickness=0)
        self.canvas.pack(pady=10)
        self.draw_idle_wave()

        self.mic_button = tk.Button(
            self.root, text="Loading Engine...", font=("Helvetica", 12, "bold"), 
            bg="#f9e2af", fg="#11111b", command=self.toggle_mic, 
            width=15, height=2, relief="flat", state="disabled"
        )
        self.mic_button.pack(pady=20)

        self.status_label = tk.Label(self.root, text="Initializing Vosk Engine...", font=("Helvetica", 10), fg="#a6adc8", bg="#1e1e2e")
        self.status_label.pack(pady=5)

        self.text_box = scrolledtext.ScrolledText(
            self.root, width=50, height=12, font=("Courier", 10), 
            bg="#313244", fg="#cdd6f4", relief="flat", state="disabled"
        )
        self.text_box.pack(padx=20, pady=5)

    def load_vosk_model(self):
        try:
            self.model = Model(lang="en-us")
            self.recognizer = KaldiRecognizer(self.model, self.RATE)
            self.vosk_loaded = True
            self.root.after(0, self.enable_ui_ready)
        except Exception as e:
            error_msg = str(e)
            self.root.after(0, lambda: self.status_label.config(text=f"Model Load Failed: {error_msg}", fg="#f38ba8"))

    def enable_ui_ready(self):
        self.mic_button.config(text="Turn Mic ON", bg="#a6e3a1", state="normal")
        self.status_label.config(text="Vosk Ready & Awaiting Connection", fg="#a6adc8")

    def log_message(self, message):
        self.text_box.config(state="normal")
        self.text_box.insert(tk.END, message + "\n")
        self.text_box.see(tk.END)
        self.text_box.config(state="disabled")

    def draw_idle_wave(self):
        self.canvas.delete("all")
        self.canvas.create_line(0, 50, 400, 50, fill="#45475a", width=2)

    def animate_voice_vibrations(self):
        if not self.animation_running:
            self.draw_idle_wave()
            return
        self.canvas.delete("all")
        points = []
        for x in range(0, 401, 5):
            y = 50 + 20 * math.sin(x * 0.05 + self.anim_step) * math.cos(x * 0.02 + self.anim_step * 0.5)
            points.append((x, y))
        self.canvas.create_line(points, fill="#89b4fa", width=3, smooth=True)
        self.anim_step += 0.2
        self.root.after(30, self.animate_voice_vibrations)

    def toggle_mic(self):
        if not self.is_listening:
            self.is_listening = True
            self.animation_running = True
            self.mic_button.config(text="Turn Mic OFF", bg="#f38ba8")
            self.status_label.config(text="Listening...", fg="#89b4fa")
            self.animate_voice_vibrations()
            threading.Thread(target=self.audio_capture_stream, daemon=True).start()
            threading.Thread(target=self.transcription_worker, daemon=True).start()
        else:
            self.reset_mic_state()

    def audio_capture_stream(self):
        try:
            stream = self.pyaudio_instance.open(
                format=self.FORMAT, channels=self.CHANNELS,
                rate=self.RATE, input=True, frames_per_buffer=self.CHUNK
            )
            while self.is_listening:
                try:
                    data = stream.read(2048, exception_on_overflow=False)
                    self.audio_queue.put(data)
                except IOError:
                    pass
            stream.stop_stream()
            stream.close()
        except Exception as e:
            self.root.after(0, lambda: self.log_message(f"[Mic Error] {str(e)}"))

    def transcription_worker(self):
        while self.is_listening:
            try:
                data = self.audio_queue.get(timeout=1)
                if self.recognizer.AcceptWaveform(data):
                    result = json.loads(self.recognizer.Result())
                    user_text = result.get("text", "").strip()
                    if user_text:
                        self.root.after(0, lambda t=user_text: self.log_message(f"🗣️ Transcribed: {t}"))
                        self.forward_to_server(user_text)
                else:
                    partial_result = json.loads(self.recognizer.PartialResult())
                    partial_text = partial_result.get("partial", "").strip()
                    if partial_text:
                        self.root.after(0, lambda t=partial_text: self.status_label.config(text=f"🗣️ {t}...", fg="#a6e3a1"))
            except queue.Empty:
                continue

    def forward_to_server(self, text):
        try:
            response = requests.post(self.server_url, json={"text": text}, timeout=5)
            if response.status_code == 200:
                ai_reply = response.json().get("ai_response", "")
                self.root.after(0, lambda r=ai_reply: self.log_message(f"🧠 Joule Response Route: {r}"))
        except Exception as e:
            self.root.after(0, lambda e_str=str(e): self.log_message(f"[Server Delivery Failed]: {e_str}"))

    def reset_mic_state(self):
        self.is_listening = False
        self.animation_running = False
        with self.audio_queue.mutex:
            self.audio_queue.queue.clear()
        self.mic_button.config(text="Turn Mic ON", bg="#a6e3a1")
        self.status_label.config(text="System Standby", fg="#a6adc8")

if __name__ == "__main__":
    root = tk.Tk()
    app = JouleVttApp(root)
    root.mainloop()