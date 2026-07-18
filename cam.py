import cv2
import requests
import time
import threading

# Configuration
# 1. Swap this with the exact IP Webcam address displayed on your phone screen
IP_WEBCAM_URL = "https://192.168.1.8:8080/video"

# 2. The endpoint on your gemini_server.py handling object data matrices
SERVER_URL = "http://127.0.0.1:5000/api/yolo_update"

def mock_object_detector(frame):
    """
    Simulated object detector.
    For a hackathon demo, this simulates detecting items present in front of the lens.
    You can swap this section out with real YOLOv8/Ultralytics code later if time permits!
    """
    # Simply looking at frame metrics to pretend it's scanning different things
    detected = ["desk", "light"]
    
    # Just a simple hackathon simulation helper:
    # If the camera detects high activity or lighting changes, change the detected object array context
    avg_color = frame.mean()
    if avg_color > 120:
        detected.extend(["apple", "banana", "healthy breakfast options"])
    else:
        detected.extend(["smartphone", "textbook", "study notes"])
        
    return list(set(detected))

def tracking_loop():
    print(f"Connecting to IP Webcam stream at: {IP_WEBCAM_URL}")
    cap = cv2.VideoCapture(IP_WEBCAM_URL)
    
    if not cap.isOpened():
        print("[Error] Could not open video stream. Verify your phone's IP address and WiFi connection.")
        return

    print("Stream connected successfully! Sending tracked updates to Flask backend...")
    
    last_update_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Warning] Dropped frame from network stream. Retrying...")
            time.sleep(0.5)
            continue
            
        # Optional local window check to ensure your computer sees the feed
        # Resize frame slightly so it doesn't take up your whole desktop screen
        display_frame = cv2.resize(frame, (640, 480))
        cv2.putText(display_frame, "Joule Vision Engine Running", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (161, 227, 166), 2)
        cv2.imshow("Vision Processing Node", display_frame)
        
        # Run our frame scanner function
        detected_items = mock_object_detector(frame)
        
        # Throttle server updates to once every 2 seconds so we don't choke the network
        if time.time() - last_update_time > 2.0:
            try:
                payload = {"detected_objects": detected_items}
                # Fire the updated tracking matrix over to the central engine
                requests.post(SERVER_URL, json=payload, timeout=2)
                print(f"[Vision Update Sent]: {detected_items}")
            except Exception as e:
                print(f"[Network Sync Warning] Backend unreachable: {e}")
            last_update_time = time.time()

        # Press 'q' inside the OpenCV window screen to quit safely
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    tracking_loop()