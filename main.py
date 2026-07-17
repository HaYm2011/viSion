import argparse
import logging
import time
import uvicorn


from src.db.database import init_db
from src.vision.tracker import VisionTracker
from src.voice.audio_loop import VoiceAssistant
from src.llm.router import get_answer
from src.api.server import app

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_vision():
    logger.info("Starting Vision Loop...")
    tracker = VisionTracker()

    # Simple mocked loop
    # In a real scenario, this would read from cv2.VideoCapture
    for i in range(5):
        logger.info(f"Processing frame {i}")
        # Mock frame processing
        tracker.process_frame(frame=None)
        time.sleep(1)
    logger.info("Vision Loop ended.")

def run_voice():
    logger.info("Starting Voice Loop...")
    assistant = VoiceAssistant()
    assistant.run_loop(router_callback=get_answer)
    logger.info("Voice Loop ended.")

def run_api():
    logger.info("Starting API Server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Object Memory Ambient Assistant")
    parser.add_argument("--mode", choices=["vision", "voice", "api", "init_db"], required=True, help="Mode to run the application in")

    args = parser.parse_args()

    if args.mode == "init_db":
        init_db()
        logger.info("Database initialized.")
    elif args.mode == "vision":
        # Ensure DB is ready
        init_db()
        run_vision()
    elif args.mode == "voice":
        run_voice()
    elif args.mode == "api":
        run_api()
