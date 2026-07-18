"""
viSion — single-file ambient object memory assistant
=====================================================
pip install ultralytics fastapi "uvicorn[standard]" opencv-python faster-whisper pyttsx3 sounddevice numpy scipy python-dotenv requests

Run:
    python main.py

Then open http://localhost:8000 in Chrome.
Voice: hold the button in the browser (Web Speech API, zero install), or say
"hey vision" hands-free once the server-side mic loop is listening.

General knowledge: copy .env.example to .env and set GEMINI_API_KEY to route
any question that isn't about a tracked object to Gemini instead.
"""

import base64, json, math, os, queue, re, threading, time, wave
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pyttsx3
import requests
import sounddevice as sd
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from faster_whisper import WhisperModel
from scipy.signal import resample
from ultralytics import YOLOE

load_dotenv()

# ══════════════════════════════════════════════════════════════════════
#  CONFIG — tweak these, nothing else
# ══════════════════════════════════════════════════════════════════════
CAM_INDEX     = 0          # 0 = default webcam
INFER_EVERY   = 4          # run YOLOE every Nth frame; display runs every frame
CONF          = 0.22       # detection confidence threshold
DEBOUNCE      = 3          # detections before we trust a zone assignment
MISSING_AFTER = 10.0       # seconds unseen → mark missing
WAKE_PHRASES  = ["hey vision", "hey vishen", "hey vison", "ok vision"]
WHISPER_MODEL = os.getenv("VISION_WHISPER", "base.en")
MODEL         = os.getenv("VISION_MODEL", "yoloe-11m-seg.pt")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyDEUgyq5ruZz8Tq7-XCOi0qiar_vv_uxgc")
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# Objects to track — open-vocab (YOLOE), so this can be broad.
TRACKED = [
    "keys", "wallet", "mobile phone", "water bottle", "spectacles",
    "backpack", "laptop", "pen", "wristwatch", "wireless mouse",
    "headphones", "charger cable", "mug", "notebook", "remote control",
    "earbuds case", "sunglasses", "umbrella", "id card", "coin purse",
    "hair brush", "scissors", "stapler", "tape dispenser", "medicine bottle",
    "power bank", "flash drive", "book", "tissue box", "keychain",
]

ALIASES: dict[str, list[str]] = {
    "keys":            ["key", "car keys", "chaabi", "chabi", "house key"],
    "wallet":          ["purse", "billfold", "money"],
    "mobile phone":    ["phone", "cell", "mobile", "smartphone", "iphone", "android"],
    "water bottle":    ["bottle", "sipper", "flask"],
    "spectacles":      ["glasses", "specs", "chashma", "eyeglasses"],
    "backpack":        ["bag", "rucksack", "knapsack", "school bag"],
    "laptop":          ["computer", "mac", "macbook", "notebook computer"],
    "pen":             ["pencil", "marker", "sketch pen"],
    "wristwatch":      ["watch"],
    "wireless mouse":  ["mouse"],
    "headphones":      ["earphones", "headset", "airpods"],
    "charger cable":   ["charger", "cable", "wire", "usb cable"],
    "mug":             ["cup", "glass"],
    "notebook":        ["diary", "notepad", "journal"],
    "remote control":  ["remote", "tv remote", "ac remote"],
    "earbuds case":    ["earbuds", "airpod case"],
    "sunglasses":      ["shades", "goggles"],
    "umbrella":        ["chatri"],
    "id card":         ["identity card", "badge", "access card"],
    "coin purse":      ["change purse"],
    "hair brush":      ["comb", "brush"],
    "scissors":        ["cutter", "scissor"],
    "stapler":         [],
    "tape dispenser":  ["tape"],
    "medicine bottle": ["pills", "tablets", "medicine"],
    "power bank":      ["powerbank", "battery pack"],
    "flash drive":     ["pen drive", "usb drive", "thumb drive"],
    "book":            ["textbook", "novel"],
    "tissue box":      ["tissues", "napkins"],
    "keychain":        [],
}

# Surfaces/places YOLOE also looks for each frame — an object's "zone" is
# whichever detected place box contains it. No manual zone rectangles.
PLACES = ["desk", "table", "shelf", "bed", "couch", "chair", "floor", "cabinet", "drawer"]
CONTAINERS = {"drawer", "cabinet"}   # "in" not "on"; going missing here means stored, not lost
ACTORS = ["person"]                  # detected but not a place/object — used to spot held items

CROPS_DIR = Path("crops"); CROPS_DIR.mkdir(exist_ok=True)

# ══════════════════════════════════════════════════════════════════════
#  SHARED STATE  (all writes go through _lock)
# ══════════════════════════════════════════════════════════════════════
_lock       = threading.Lock()
_state:  dict[str, dict] = {}          # label → {zone, status, conf, seen_at, crop}
_events: deque            = deque(maxlen=500)
_jpeg:   list             = [None]      # latest camera frame as JPEG bytes
_pending: dict[str, deque] = defaultdict(lambda: deque(maxlen=DEBOUNCE))
_zones_now: list          = [[]]        # currently-detected place names (shared with /state)
_voice:  dict             = {"listening": False, "heard": "", "say": "", "hit": None}
_speak_queue: queue.Queue = queue.Queue()   # text handed to the TTS thread — Q&A answers + event narration
_tts_busy = threading.Event()               # set while audio is actually playing, so the mic ignores its own echo

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

def _ago(iso: str) -> str:
    secs = (datetime.now(timezone.utc) - datetime.fromisoformat(iso)).total_seconds()
    if secs < 90:   return "just now"
    if secs < 3600: return f"{int(secs // 60)} min ago"
    return f"{int(secs // 3600)} hr ago"

def _iou(a: tuple, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    if inter == 0:
        return 0.0
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0

def _dedupe_overlaps(dets: list, iou_thresh: float = 0.5) -> list:
    """A single physical object can get proposed as several different labels
    at once (worse from odd camera angles) — keep only the most confident."""
    kept: list = []
    for d in sorted(dets, key=lambda d: -d[1]):
        if not any(_iou(d[2:], k[2:]) > iou_thresh for k in kept):
            kept.append(d)
    return kept

def _is_held(box: tuple, person_boxes: list, thresh: float = 0.35) -> bool:
    x1, y1, x2, y2 = box
    obj_area = max(0, x2 - x1) * max(0, y2 - y1)
    if obj_area == 0:
        return False
    for p in person_boxes:
        iw = max(0, min(x2, p[2]) - max(x1, p[0]))
        ih = max(0, min(y2, p[3]) - max(y1, p[1]))
        if (iw * ih) / obj_area > thresh:
            return True
    return False

def _zone_base(zone: str) -> str:
    """Strip a numbering suffix like "drawer 2" -> "drawer" for CONTAINERS checks."""
    head, _, tail = zone.rpartition(" ")
    return head if head and tail.isdigit() else zone

def _zone_phrase(zone: str) -> str:
    if zone == "your hand":
        return "in your hand"
    return f"{'in' if _zone_base(zone) in CONTAINERS else 'on'} the {zone}"

# ══════════════════════════════════════════════════════════════════════
#  VISION LOOP
# ══════════════════════════════════════════════════════════════════════
def _vision_loop():
    print(f"[vision] loading {MODEL} …")
    model = YOLOE(MODEL)
    all_classes = PLACES + TRACKED + ACTORS
    model.set_classes(all_classes, model.get_text_pe(all_classes))
    places_set, tracked_set, actors_set = set(PLACES), set(TRACKED), set(ACTORS)
    print(f"[vision] ready — places: {', '.join(PLACES)}")
    print(f"[vision] ready — tracking: {', '.join(TRACKED)}")

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print(f"[vision] CAM_INDEX={CAM_INDEX} unavailable, scanning for a camera …")
        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                print(f"[vision] using camera index {i} instead")
                break
        else:
            print("[vision] no camera found — vision loop idle")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)

    n = 0
    last_boxes  = []   # persist overlay between infer frames
    last_places = []   # [(name, x1, y1, x2, y2)]

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05); continue
        n += 1
        H, W = frame.shape[:2]

        if n % INFER_EVERY == 0:
            results   = model.predict(frame, conf=CONF, verbose=False)[0]
            last_boxes = results.boxes

            raw_places: dict[str, list[tuple[int, int, int, int]]] = defaultdict(list)
            object_dets: list[tuple[str, float, int, int, int, int]] = []
            person_boxes: list[tuple[int, int, int, int]] = []
            for box in last_boxes:
                label = model.names[int(box.cls)]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if label in places_set:
                    raw_places[label].append((x1, y1, x2, y2))
                elif label in tracked_set:
                    object_dets.append((label, float(box.conf), x1, y1, x2, y2))
                elif label in actors_set:
                    person_boxes.append((x1, y1, x2, y2))
            object_dets = _dedupe_overlaps(object_dets)

            # Number same-label places left-to-right (e.g. "drawer 1", "drawer 2")
            # so multiple instances stay distinguishable. No persistent tracker
            # needed — furniture doesn't move, so spatial order is stable.
            place_boxes: list[tuple[str, int, int, int, int]] = []
            for label, boxes in raw_places.items():
                uniq: list[tuple[int, int, int, int]] = []
                for b in boxes:                       # collapse duplicate proposals
                    if not any(_iou(b, u) > 0.6 for u in uniq):
                        uniq.append(b)
                if len(uniq) > 1:
                    uniq.sort(key=lambda b: b[0])      # left → right by x1
                    for i, b in enumerate(uniq, start=1):
                        place_boxes.append((f"{label} {i}", *b))
                else:
                    place_boxes.append((label, *uniq[0]))
            last_places = place_boxes

            with _lock:
                _zones_now[0] = sorted({name for name, *_ in place_boxes})

            seen_labels: set[str] = set()
            for label, conf, x1, y1, x2, y2 in object_dets:
                if _is_held((x1, y1, x2, y2), person_boxes):
                    zone = "your hand"
                else:
                    # Use bottom-centre as the "resting point"
                    cx, cy = (x1 + x2) // 2, y2
                    zone, best_area = None, None
                    for name, px1, py1, px2, py2 in place_boxes:
                        if px1 <= cx <= px2 and py1 <= cy <= py2:
                            area = (px2 - px1) * (py2 - py1)
                            if best_area is None or area < best_area:
                                best_area, zone = area, name
                    zone = zone or "floor"   # no surface detected under it → assume floor
                seen_labels.add(label)

                # Debounce: only trust a zone once N consecutive reads agree
                _pending[label].append(zone)
                if len(_pending[label]) < DEBOUNCE or len(set(_pending[label])) > 1:
                    continue

                with _lock:
                    prev = _state.get(label)

                if prev and prev["zone"] == zone and prev["status"] == "present":
                    with _lock: _state[label]["seen_at"] = _now()
                    continue   # no real transition, skip

                # Save crop (padded, clamped)
                pad = 14
                crop = frame[max(0, y1-pad):min(H, y2+pad),
                             max(0, x1-pad):min(W, x2+pad)]
                fname = f"{label.replace(' ','_')}_{int(time.time())}.jpg"
                cv2.imwrite(str(CROPS_DIR / fname), crop)

                event = "appeared" if not prev else "moved"
                with _lock:
                    _state[label] = {"zone": zone, "status": "present",
                                     "conf": round(conf, 3),
                                     "seen_at": _now(), "crop": fname}
                    _events.append({"object": label, "event": event, "zone": zone,
                                    "conf": round(conf, 3), "at": _now()})
                print(f"[event] {label:16s} {event:10s} → {zone}")
                verb = "just appeared" if event == "appeared" else "moved"
                _speak_queue.put(f"Your {label} {verb} {_zone_phrase(zone)}.")

            # Flip unseen objects to missing (or "stored", if last seen in a
            # container) after timeout, but KEEP their zone.
            with _lock:
                for label, s in _state.items():
                    if label in seen_labels or s["status"] in ("missing", "stored"):
                        continue
                    age = (datetime.now(timezone.utc)
                           - datetime.fromisoformat(s["seen_at"])).total_seconds()
                    if age > MISSING_AFTER:
                        stored = _zone_base(s["zone"]) in CONTAINERS
                        s["status"] = "stored" if stored else "missing"
                        event = "stored" if stored else "disappeared"
                        _events.append({"object": label, "event": event,
                                        "zone": s["zone"], "conf": s["conf"],
                                        "at": _now()})
                        print(f"[event] {label:16s} {event:10s} → {s['zone']}")
                        if stored:   # only narrate "put away", not every ordinary timeout-to-missing
                            _speak_queue.put(f"Your {label} was just stored {_zone_phrase(s['zone'])}.")

        # ── draw overlay ──────────────────────────────────────────────
        vis = frame.copy()
        # Detected places
        for name, x1, y1, x2, y2 in last_places:
            cv2.rectangle(vis, (x1, y1), (x2, y2), (140, 145, 130), 1)
            cv2.putText(vis, name.upper(), (x1 + 7, y1 + 19),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (140, 145, 130), 1)
        # Detected objects
        for box in last_boxes:
            lbl = model.names[int(box.cls)]
            if lbl not in tracked_set:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(vis, (x1, y1), (x2, y2), (45, 130, 90), 2)
            cv2.putText(vis, lbl, (x1, max(y1-6, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (45, 130, 90), 2)

        _jpeg[0] = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 72])[1].tobytes()

# ══════════════════════════════════════════════════════════════════════
#  QUERY RESOLVER  (no LLM needed — pure string match)
# ══════════════════════════════════════════════════════════════════════
def resolve(q: str) -> str | None:
    q = q.lower()
    # Longest matching phrase (label or alias) wins — avoids a short label
    # like "pen" shadowing a longer alias like "pen drive" (→ flash drive).
    best: tuple[int, str] | None = None
    for label in TRACKED:
        for phrase in [label, *ALIASES.get(label, [])]:
            if re.search(rf"\b{re.escape(phrase)}\b", q) and (best is None or len(phrase) > best[0]):
                best = (len(phrase), label)
    if best:
        return best[1]
    for label in TRACKED:                        # word-level fallback
        if any(w in q.split() for w in label.split()):
            return label
    return None

def ask_gemini(q: str) -> str:
    """Anything that isn't about a tracked object goes here instead."""
    if not GEMINI_API_KEY:
        return ("I'm not tracking that, and no Gemini key is set — "
                "add GEMINI_API_KEY to .env for general questions.")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    body = {
        "contents": [{"parts": [{"text": q}]}],
        "systemInstruction": {"parts": [{
            "text": "You are viSion, a voice assistant. Answer in 1-3 short sentences "
                    "suitable for being read aloud."
        }]},
    }
    try:
        r = requests.post(url, params={"key": GEMINI_API_KEY}, json=body, timeout=15)
        r.raise_for_status()
        return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as e:
        print(f"[gemini] error: {e}")
        return "Sorry, I couldn't reach Gemini just now."

def answer(q: str) -> dict:
    """Shared by /ask and the voice loop so both give identical responses."""
    label = resolve(q)
    if not label:
        return {"say": ask_gemini(q), "hit": None}

    with _lock:
        s = _state.get(label)
    if not s:
        return {"say": f"I haven't spotted your {label} yet.", "hit": None}

    ago = _ago(s["seen_at"])
    if s["status"] == "present":
        say = f"Your {label} is {_zone_phrase(s['zone'])}."
    elif s["status"] == "stored":
        say = f"Your {label} is stored {_zone_phrase(s['zone'])} — put there {ago}."
    else:
        say = f"I last saw your {label} {_zone_phrase(s['zone'])}, {ago}."

    return {"say": say, "hit": {"label": label, **s, "ago": ago,
                                "container": _zone_base(s["zone"]) in CONTAINERS}}

# ══════════════════════════════════════════════════════════════════════
#  VOICE LOOP  (hands-free — server-side mic, wake word, TTS)
# ══════════════════════════════════════════════════════════════════════
SAMPLE_RATE  = 16000
CHUNK_S      = 0.25         # seconds per audio chunk
SILENCE_RMS  = 0.010        # below this = silence
SILENCE_HANG = 8            # consecutive silent chunks that end an utterance
MAX_UTTER_S  = 8            # hard cap on one utterance's length

def _tts_loop():
    """Owns the one pyttsx3 engine — everything that wants to talk (the
    wake-word Q&A flow, ambient event narration) hands text to _speak_queue
    instead of calling pyttsx3 directly, so only one utterance plays at a time.

    pyttsx3's Linux driver plays audio via a bare `aplay file.wav` call with
    no device flag, which fights PipeWire for the raw hardware PCM and fails
    with "Device or resource busy". Route around it: synthesize to a file,
    then play that file ourselves through sounddevice/PortAudio, which
    already talks to PipeWire fine (same path the mic capture uses)."""
    tts = pyttsx3.init()
    tts.setProperty("rate", 175)
    tmp_wav = f"/tmp/vision_tts_{os.getpid()}.wav"
    try:
        out_sr = int(sd.query_devices(kind="output")["default_samplerate"])
    except Exception:
        out_sr = 44100

    while True:
        text = _speak_queue.get()
        with _lock:
            _voice["say"] = text
        _tts_busy.set()
        try:
            tts.save_to_file(text, tmp_wav)
            tts.runAndWait()
            with wave.open(tmp_wav, "rb") as wf:
                sr, ch = wf.getframerate(), wf.getnchannels()
                raw = wf.readframes(wf.getnframes())
            data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            if ch > 1:
                data = data.reshape(-1, ch)
            if sr != out_sr:
                data = resample(data, int(len(data) * out_sr / sr)).astype(np.float32)
            sd.play(data, out_sr)
            sd.wait()
        except Exception as e:
            print(f"[voice] tts error: {e}")
        time.sleep(0.3)   # let room/speaker echo settle before the mic listens again
        _tts_busy.clear()

def _voice_loop():
    print(f"[voice] loading whisper ({WHISPER_MODEL}) …")
    try:
        whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    except Exception as e:
        print(f"[voice] could not load whisper, voice disabled: {e}")
        return

    def transcribe(audio: np.ndarray) -> str:
        segments, _ = whisper.transcribe(audio, language="en", vad_filter=True)
        return " ".join(seg.text for seg in segments).strip().lower()

    chunk_n = int(SAMPLE_RATE * CHUNK_S)
    audio_q: queue.Queue = queue.Queue()

    def callback(indata, frames, time_info, status):
        if not _tts_busy.is_set():   # ignore the mic while we're talking (no self-echo)
            audio_q.put(indata[:, 0].copy())

    try:
        stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                                 blocksize=chunk_n, callback=callback)
    except Exception as e:
        print(f"[voice] no microphone available, voice disabled: {e}")
        return

    print(f"[voice] listening — say '{WAKE_PHRASES[0]}'")
    with stream:
        buf: list = []
        speaking, silence_run, awaiting_command = False, 0, False

        while True:
            try:
                chunk = audio_q.get()
                rms = float(np.sqrt(np.mean(chunk ** 2)))

                if rms > SILENCE_RMS:
                    buf.append(chunk); speaking = True; silence_run = 0
                elif speaking:
                    buf.append(chunk); silence_run += 1

                utter_done = speaking and (silence_run >= SILENCE_HANG or
                                            len(buf) * CHUNK_S >= MAX_UTTER_S)
                if not utter_done:
                    continue

                audio = np.concatenate(buf)
                buf, speaking, silence_run = [], False, 0

                text = transcribe(audio)
                # Drop whatever queued up while we were transcribing (avoids stale audio)
                while not audio_q.empty():
                    audio_q.get_nowait()
                if not text:
                    continue
                print(f"[voice] heard: {text!r}")
                with _lock: _voice["heard"] = text

                if awaiting_command:
                    awaiting_command = False
                    with _lock: _voice["listening"] = False
                    d = answer(text)
                    with _lock: _voice["hit"] = d["hit"]
                    _speak_queue.put(d["say"])
                else:
                    phrase = next((w for w in WAKE_PHRASES if w in text), None)
                    if not phrase:
                        continue
                    rest = text.split(phrase, 1)[1].strip()
                    if rest:
                        d = answer(rest)
                        with _lock: _voice["hit"] = d["hit"]
                        _speak_queue.put(d["say"])
                    else:
                        awaiting_command = True
                        with _lock: _voice["listening"] = True
                        _speak_queue.put("Yes?")
            except Exception as e:
                print(f"[voice] loop error: {e}")
                buf, speaking, silence_run = [], False, 0

# ══════════════════════════════════════════════════════════════════════
#  FASTAPI  — every route in one place
# ══════════════════════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app):
    threading.Thread(target=_vision_loop, daemon=True).start()
    threading.Thread(target=_tts_loop, daemon=True).start()
    threading.Thread(target=_voice_loop, daemon=True).start()
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/", response_class=HTMLResponse)
def index():
    return HTML   # defined at bottom of file

@app.get("/feed")
def feed():
    def gen():
        while True:
            if _jpeg[0]:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + _jpeg[0] + b"\r\n"
            time.sleep(0.04)
    return StreamingResponse(gen(),
                             media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/crops/{name}")
def crop(name: str):
    return FileResponse(CROPS_DIR / name)

@app.get("/state")
def state():
    with _lock:
        objs   = [{"label": k, **v} for k, v in _state.items()]
        recent = list(_events)[-25:][::-1]
        zones  = list(_zones_now[0])
        voice  = dict(_voice)
    return JSONResponse({"objects": objs,
                         "events":  recent,
                         "zones":   zones,
                         "voice":   voice})

@app.get("/ask")
def ask(q: str = ""):
    return answer(q)

# ══════════════════════════════════════════════════════════════════════
#  SINGLE-FILE HTML  (served from memory, no separate file needed)
# ══════════════════════════════════════════════════════════════════════
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>viSion</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Archivo:wght@500;700&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#DDE1D8;--bg2:#D2D7CB;--ink:#1A1F1C;--muted:#6B7268;
  --rule:#B4BAAE;--green:#2F5D50;--red:#9E2B25;--amber:#9E7B25;
}
body{
  background:var(--bg);color:var(--ink);
  font:400 14px/1.55 "IBM Plex Mono",monospace;
  padding:24px 20px 60px;
  background-image:repeating-linear-gradient(var(--bg) 0 27px,var(--rule) 27px 28px);
}
.wrap{max-width:1100px;margin:0 auto}
header{display:flex;align-items:baseline;gap:12px;border-bottom:2px solid var(--ink);padding-bottom:10px;margin-bottom:22px}
h1{font:700 26px/1 "Archivo",sans-serif;letter-spacing:-.02em;text-transform:uppercase}
.sub{font-size:11px;color:var(--muted);letter-spacing:.16em;text-transform:uppercase}
.count{margin-left:auto;font-size:12px;color:var(--muted)}
.grid{display:grid;grid-template-columns:1.2fr .8fr;gap:22px}
@media(max-width:820px){.grid{grid-template-columns:1fr}}
.lbl{font:500 10px/1 "Archivo",sans-serif;letter-spacing:.18em;text-transform:uppercase;
     color:var(--muted);margin-bottom:8px}
.feed-wrap{border:1px solid var(--ink);background:#000;aspect-ratio:16/9;overflow:hidden}
.feed-wrap img{width:100%;height:100%;object-fit:cover;display:block}
.ask-row{display:flex;gap:8px;margin-top:14px}
button{font:700 12px/1 "Archivo",sans-serif;letter-spacing:.1em;text-transform:uppercase;
       border:1px solid var(--ink);background:var(--ink);color:var(--bg);
       padding:13px 18px;cursor:pointer;white-space:nowrap;transition:transform .06s}
button:active{transform:translateY(2px)}
button.on{background:var(--red);border-color:var(--red);animation:pulse 1s infinite}
@keyframes pulse{50%{opacity:.6}}
@media(prefers-reduced-motion:reduce){button.on{animation:none}}
input{flex:1;border:1px solid var(--rule);background:var(--bg);
      padding:0 12px;font:400 13px "IBM Plex Mono",monospace;color:var(--ink)}
input:focus{outline:2px solid var(--green);outline-offset:-2px}
.heard{font-size:12px;color:var(--muted);margin-top:8px;min-height:1.3em}
.voice-status{font-size:11px;color:var(--muted);margin-top:6px;display:flex;align-items:center;gap:6px}
.voice-status .vdot{width:6px;height:6px;border-radius:50%;background:var(--muted);flex:none}
.voice-status.listening .vdot{background:var(--red);animation:pulse 1s infinite}
.voice-status.listening{color:var(--red)}
.receipt{border:1px solid var(--ink);background:var(--bg2);
         padding:18px;min-height:220px;position:relative;display:flex;
         flex-direction:column;gap:12px}
.receipt .idle{color:var(--muted);margin:auto;text-align:center;
               max-width:24ch;line-height:1.8;font-size:13px}
.rec-top{display:flex;gap:16px;align-items:flex-start}
.polaroid{width:120px;flex:none;background:#fff;padding:7px 7px 22px;
          border:1px solid var(--rule);transform:rotate(-2deg);
          box-shadow:2px 3px 0 rgba(0,0,0,.12)}
.polaroid img{width:100%;display:block;aspect-ratio:1;object-fit:cover;background:#222}
.rec-body h2{font:700 22px/1.1 "Archivo",sans-serif;text-transform:uppercase}
.rec-body .where{font-size:18px;margin-top:6px}
.rec-body .where b{border-bottom:2px solid var(--green)}
.rec-body .ts{font-size:11px;color:var(--muted);margin-top:10px}
.stamp{position:absolute;top:14px;right:14px;font:700 11px/1 "Archivo",sans-serif;
       letter-spacing:.2em;text-transform:uppercase;border:2px solid;
       padding:5px 9px;transform:rotate(4deg)}
.stamp.present{color:var(--green);border-color:var(--green)}
.stamp.missing{color:var(--red);border-color:var(--red)}
.stamp.stored{color:var(--amber);border-color:var(--amber)}
table{width:100%;border-collapse:collapse;margin-top:6px}
th{font:500 9px/1 "Archivo",sans-serif;letter-spacing:.16em;text-transform:uppercase;
   color:var(--muted);text-align:left;padding:0 0 7px;border-bottom:1px solid var(--ink)}
td{padding:8px 0;border-bottom:1px solid var(--rule);font-size:12px;vertical-align:middle}
.dot{display:inline-block;width:7px;height:7px;border-radius:50%;margin-right:7px;vertical-align:1px}
.dot.present{background:var(--green)} .dot.missing{background:var(--red)} .dot.stored{background:var(--amber)}
.empty{color:var(--muted);padding:18px 0;font-size:12px}
.log{margin-top:22px}
.log li{list-style:none;display:flex;gap:10px;font-size:11px;
        padding:5px 0;border-bottom:1px dotted var(--rule)}
.log .t{color:var(--muted);flex:none;width:65px}
.log .e{flex:none;width:90px;text-transform:uppercase;font-size:9px;
        letter-spacing:.1em;font-family:"Archivo",sans-serif;font-weight:700;padding-top:2px}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>viSion</h1>
    <span class="sub">object memory</span>
    <span class="count" id="cnt">—</span>
  </header>
  <div class="grid">
    <div>
      <div class="lbl">Live — zones outlined</div>
      <div class="feed-wrap"><img src="/feed" alt="live feed"></div>
      <div class="ask-row">
        <button id="mic">🎤 Ask</button>
        <input id="q" placeholder='where are my keys?' aria-label="Ask about an object">
      </div>
      <div class="heard" id="heard"></div>
      <div class="voice-status" id="voiceStatus"><span class="vdot"></span><span id="voiceText">say "hey vision" anytime — hands-free</span></div>
    </div>
    <div>
      <div class="lbl">Last known</div>
      <div class="receipt" id="receipt">
        <div class="idle">Ask where something is.<br>You'll get the photo + location.</div>
      </div>
      <div style="margin-top:22px">
        <div class="lbl">Index</div>
        <table><thead><tr><th>Object</th><th>Zone</th><th>Seen</th></tr></thead>
        <tbody id="rows"></tbody></table>
      </div>
      <div class="log">
        <div class="lbl">Movement log</div>
        <ul id="log"></ul>
      </div>
    </div>
  </div>
</div>
<script>
const $ = s => document.querySelector(s);
const hhmm = iso => new Date(iso).toLocaleTimeString([],{hour:'2-digit',minute:'2-digit'});

function speak(t){
  try{speechSynthesis.cancel();
    const u=new SpeechSynthesisUtterance(t);u.lang='en-IN';u.rate=1.05;
    speechSynthesis.speak(u);}catch(e){}
}

function whereText(h){
  const prep = h.container ? 'in' : 'on';
  if(h.status==='present') return prep+' the';
  if(h.status==='stored')  return 'stored '+prep+' the';
  return 'last '+prep+' the';
}
function stampText(status){
  return status==='present' ? 'On record' : status==='stored' ? 'Stored away' : 'Missing';
}

function renderReceipt(say, hit){
  const box=$('#receipt');
  if(!hit){box.innerHTML='<div class="idle">'+say+'</div>';return;}
  const h=hit;
  box.innerHTML=`
    <div class="stamp ${h.status}">${stampText(h.status)}</div>
    <div class="rec-top">
      <figure class="polaroid"><img src="/crops/${h.crop}" alt="crop"></figure>
      <div class="rec-body">
        <h2>${h.label}</h2>
        <div class="where">${whereText(h)} <b>${h.zone}</b></div>
        <div class="ts">${hhmm(h.seen_at)} · ${h.ago} · ${h.conf} conf</div>
      </div>
    </div>`;
}

async function ask(q){
  if(!q.trim())return;
  $('#heard').textContent = '"' + q + '"';
  const d = await fetch('/ask?q='+encodeURIComponent(q)).then(r=>r.json());
  speak(d.say);
  renderReceipt(d.say, d.hit);
}

// Browser speech
const SR=window.SpeechRecognition||window.webkitSpeechRecognition;
if(SR){
  const rec=new SR();rec.lang='en-IN';rec.interimResults=false;
  rec.onresult=e=>ask(e.results[0][0].transcript);
  rec.onend=()=>$('#mic').classList.remove('on');
  rec.onerror=e=>$('#heard').textContent='Mic error: '+e.error+'. Type instead.';
  const go=()=>{$('#mic').classList.add('on');try{rec.start();}catch(e){}};
  $('#mic').addEventListener('mousedown',go);
  $('#mic').addEventListener('touchstart',e=>{e.preventDefault();go();});
  $('#mic').addEventListener('mouseup',()=>rec.stop());
  $('#mic').addEventListener('touchend',()=>rec.stop());
}else{
  $('#mic').disabled=true;$('#mic').title='Needs Chrome';
}
$('#q').addEventListener('keydown',e=>{
  if(e.key==='Enter'){ask(e.target.value);e.target.value='';}
});

// Poll state
async function poll(){
  try{
    const d=await fetch('/state').then(r=>r.json());
    $('#cnt').textContent=d.objects.length+' tracked · '+d.zones.length+' zones';
    const rows=d.objects.sort((a,b)=>b.seen_at.localeCompare(a.seen_at));
    $('#rows').innerHTML=rows.length
      ?rows.map(o=>`<tr>
          <td><span class="dot ${o.status}"></span>${o.label}</td>
          <td>${o.zone}</td><td>${hhmm(o.seen_at)}</td></tr>`).join('')
      :'<tr><td colspan="3" class="empty">Nothing seen yet — put something in frame.</td></tr>';
    $('#log').innerHTML=d.events.map(e=>`
      <li><span class="t">${hhmm(e.at)}</span>
          <span class="e">${e.event}</span>
          <span>${e.object} — ${e.zone}</span></li>`).join('');

    const v=d.voice, vs=$('#voiceStatus');
    vs.classList.toggle('listening', v.listening);
    $('#voiceText').textContent = v.listening ? 'listening for your question…'
      : (v.heard ? 'heard: "'+v.heard+'"' : 'say "hey vision" anytime — hands-free');
    if(v.heard && v.heard!==lastVoiceHeard){
      lastVoiceHeard=v.heard;
      $('#heard').textContent='🎙 "'+v.heard+'"';
      renderReceipt(v.say, v.hit);
    }
  }catch(e){}
}
let lastVoiceHeard='';
poll();setInterval(poll,1200);
</script>
</body>
</html>"""

# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)