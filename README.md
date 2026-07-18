# viSion - An Object Memory Ambient Assistant

An ambient AI assistant designed to track the location of your everyday objects, log habits, and answer queries such as "Where are my keys?" through voice interaction.

## Architecture

- **Camera (RPi Cam / USB webcam)**
  - Motion trigger (only process on change to save compute).
  - VLM frame captioning + object detection (YOLO-World for open-vocab boxes).
  - Event log: `{object, location, timestamp, frame_ref}` -> SQLite + Vector Search.
- **Mic -> Wake word (openWakeWord) -> Whisper STT**
  - Query router -> search over event log -> LLM answer.
  - Piper TTS -> Speaker.

## The Concept: Object Permanence

viSion tracks onject position changes. For example: "Keys appeared at 14:32 on the counter" / "keys disappeared at 14:35".
Locations are spoken relative to labeled zones such as "desk", couch", etc.

## Models

- **Detection:** YOLO
- **Descriptions/Queries:** Moondream2 or Qwen2-VL-2B locally (Claude/GPT-4o via API for low-latency budget)
- **STT:** `faster-whisper`
- **TTS:** Piper
- **Wake word:** `hey viSion`, `vision`, etc.(it says etc. as it is adaptive)

## Privacy

APIs only used for exta information related to objects. We run all image detection locally on-device.
