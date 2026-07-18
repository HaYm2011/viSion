import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
from google.genai import types

app = Flask(__name__)
CORS(app)  # Enables cross-origin execution for Frontend HTML/JS queries

# Hardcoded Authentication Token Configuration 
API_KEY_STRING = "AQ.Ab8RN6LJCn0IW8BUv7WhHP_9TaBSzm6jyTGrmI_p1WPSOvj-Lg"
client = genai.Client(api_key=API_KEY_STRING)

# Local Storage State
latest_tracking_frame_data = "No tracking information received from webcam stream yet."
latest_joule_output = "Hello! I am ready to track your environment and focus metrics."

JOULE_CONTEXT_INSTRUCTION = """
You are Joule, an empathetic smart home companion. 
You are given text prompts, a summary of tracked objects from a live camera stream (like phones, books, toys), and food monitoring commands.
Your goals:
1. Locate items based on the tracking context.
2. If food is analyzed, provide standard macronutrient breakdowns (Proteins, Carbs, Fats) and micronutrient metrics.
3. Keep focus tracking parameters supportive and high-energy. Keep responses concise for immediate display.
"""

@app.route('/api/yolo_update', methods=['POST'])
def update_vision_data():
    """Receives target object arrays from your secondary camera/YOLO stream port"""
    global latest_tracking_frame_data
    data = request.get_json()
    if data and 'detected_objects' in data:
        # Expecting a structure like: {"detected_objects": ["phone", "cup", "book"]}
        objects = data['detected_objects']
        latest_tracking_frame_data = f"Currently visible objects via YOLO: {', '.join(objects)}"
        return jsonify({"status": "Tracking synchronized successfully"}), 200
    return jsonify({"error": "Malformed visualization metrics structure"}), 400

@app.route('/api/voice_input', methods=['POST'])
def receive_voice_input():
    """Receives voice conversions from VTT.py, maps context, and executes Gemini model queries"""
    global latest_joule_output, latest_tracking_frame_data
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "Missing input payload text parameter"}), 400
    
    user_prompt = data['text']
    
    # Bundle visual tracking context with the textual request
    full_prompt_payload = (
        f"Tracking Engine Data Context: {latest_tracking_frame_data}\n"
        f"User Audio Statement: {user_prompt}"
    )
    
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=full_prompt_payload,
            config=types.GenerateContentConfig(
                system_instruction=JOULE_CONTEXT_INSTRUCTION,
                temperature=0.4
            )
        )
        latest_joule_output = response.text.strip()
        return jsonify({"ai_response": latest_joule_output}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/joule_stream', methods=['GET'])
def stream_output_to_frontend():
    """Polled endpoint by your frontend JS framework to render updates"""
    global latest_joule_output
    return jsonify({
        "response": latest_joule_output
    }), 200

if __name__ == "__main__":
    print(f"Joule System Server Initialization. Operational on Loopback Port 5000.")
    app.run(host="127.0.0.1", port=5000, debug=True)