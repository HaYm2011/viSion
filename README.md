# Object Memory Ambient Assistant

An ambient AI assistant designed to track the location of your everyday objects, log habits, and answer queries like "Where are my keys?" through voice interaction.

## Architecture

* **Camera (RPi Cam / USB webcam)**
  * Motion trigger (only process on change to save compute).
  * VLM frame captioning + object detection (YOLO-World for open-vocab boxes).
  * Event log: `{object, location, timestamp, frame_ref}` -> SQLite + Vector Search.
* **Mic -> Wake word (openWakeWord) -> Whisper STT**
  * Query router -> SQL/vector search over event log -> LLM answer.
  * Piper TTS -> Speaker.

## The Core Concept: Object Permanence

The system doesn't just caption frames. It tracks state transitions: "Keys appeared at 14:32 on the counter" / "keys disappeared at 14:35". Last known location = counter.
Locations are anchored to pre-labeled zones ("counter", "desk drawer", "entryway bowl") rather than raw pixel coordinates.

## Models

* **Detection:** YOLOv11n or YOLO-World (Open-vocab).
* **Descriptions/Queries:** Moondream2 or Qwen2-VL-2B locally (Claude/GPT-4o via API for low-latency budget).
* **STT:** `faster-whisper` (base.en).
* **TTS:** Piper.
* **Wake word:** `openWakeWord`.

## Privacy (Day 1 Priorities)

1. Local-first inference. Crops sent to APIs, not full frames.
2. Hardware mute switches on camera and mic.
3. Retention policy: raw frames deleted after N hours, only event metadata and crops persist.
4. Encrypted DB at rest.

## Repository Structure

* `src/vision/` - Object detection, state transitions, and zone logic.
* `src/voice/` - Wake word, STT (faster-whisper), TTS (piper).
* `src/db/` - SQLite with vector extensions for event logging.
* `src/llm/` - Query router and LLM interface.
* `src/api/` - FastAPI backend for potential UI.
* `data/` - Local DB and crop storage.
* `config/` - Zone definitions and settings.

## Getting Started

Check out the individual modules in `src/` to get started. Build order:
1. Laptop + webcam (Vision tracking to SQLite).
2. Add zones + last_seen logic. Return crops.
3. Add voice loop.
4. Port to Raspberry Pi.
