# app.py
import os
import time
import base64
import threading
import asyncio
import uuid

import cv2
import numpy as np
import edge_tts
import openai
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from ultralytics import YOLO
import supervision as sv
from dotenv import load_dotenv, find_dotenv
from playsound import playsound

# ─── CONFIG ────────────────────────────────────────────────────────────────────
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
openai.api_key   = os.getenv("OPENAI_API_KEY")
SEND_INTERVAL    = 7        # seconds between narrations
scene_buffer     = []       # list of event dicts
buffer_lock      = threading.Lock()
last_send        = time.time()

# ─── TTS HELPERS ───────────────────────────────────────────────────────────────
async def _save_tts(text: str, outfile: str):
    com = edge_tts.Communicate(text, voice="en-GB-LibbyNeural", rate="+15%")
    await com.save(outfile)

def generate_tts_data_uri(text: str) -> str:
    """Run edge-tts in its own loop, return a data: URI of the MP3."""
    tmp_file = f"{uuid.uuid4().hex}.mp3"
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(_save_tts(text, tmp_file))
        with open(tmp_file, "rb") as f:
            b = f.read()
        uri = "data:audio/mp3;base64," + base64.b64encode(b).decode("utf-8")
    finally:
        loop.close()
        if os.path.exists(tmp_file):
            playsound(tmp_file)  # Play the sound immediately
            os.remove(tmp_file)
    return uri

def send_to_gpt():
    global last_send
    while True:
        now = time.time()
        if now - last_send >= SEND_INTERVAL:
            with buffer_lock:
                batch = scene_buffer.copy()
                scene_buffer.clear()
                last_send = now

            if batch:
                # 1) build prompt
                prompt = "You are an assistant for a visually impaired user. Here is what was just seen:\n"
                for e in batch:
                    prompt += (
                        f"- {e['class']} (ID {e['track_id']}) at "
                        f"x={e['cx']:.0f},y={e['cy']:.0f} moving "
                        f"vx={e['vx']:.1f},vy={e['vy']:.1f}\n"
                    )
                prompt += "\nPlease describe this very briefly in natural spoken language."

                # 2) call GPT-4
                try:
                    resp = openai.chat.completions.create(
                        model="gpt-4.1-2025-04-14",
                        messages=[
                            {"role": "system", "content": "You describe surroundings for a blind user."},
                            {"role": "user",   "content": prompt}
                        ],
                        temperature=0.7,
                        max_tokens=150
                    )
                    text = resp.choices[0].message.content.strip()
                except Exception as e:
                    text = f"[Error generating description: {e}]"

                # 3) generate TTS data URI
                try:
                    print(f"Generated description: {text}")
                    audio_uri = generate_tts_data_uri(text)
                except Exception as e:
                    audio_uri = None
                    text += f"\n[Audio error: {e}]"

                # 4) emit to all connected clients
                socketio.emit(
                    "narration",
                    {"text": text, "audio": audio_uri}
                )
            
        time.sleep(0.5)


# ─── FLASK / SOCKETIO SETUP ─────────────────────────────────────────────────────
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# ─── LOAD YOUR PIPELINE ─────────────────────────────────────────────────────────
model = YOLO("yolov8m-world.pt")

class_names = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
"traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird",
"cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", 
"backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
"sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
"ball", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife",
"spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
"hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
"dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
"cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
"clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush", "watch"]

model.set_classes(class_names)

box_annotator   = sv.BoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=1)

kalman_filters = {}
def create_kalman():
    kf = cv2.KalmanFilter(4,2)
    kf.transitionMatrix    = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
    kf.measurementMatrix   = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
    kf.processNoiseCov     = np.eye(4, dtype=np.float32)*1e-2
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32)*1e-1
    kf.errorCovPost        = np.eye(4, dtype=np.float32)
    return kf

# ─── ROUTES ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

# ─── WEBSOCKET HANDLER ─────────────────────────────────────────────────────────
@socketio.on("frame")
def handle_frame(msg):
    img_b64 = msg["data"].split(",",1)[1]
    arr     = np.frombuffer(base64.b64decode(img_b64), np.uint8)
    frame   = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    # run YOLO+ByteTrack
    results = model.track(
        frame,
        tracker="custom_bytetrack.yaml",
        conf=0.25,
        iou=0.5,
        persist=True,
        verbose=False
    )
    result = results[0]
    annotated = frame.copy()

    ids = result.boxes.id 
    if ids is None or len(ids) == 0:
        ids = [0] * len(result.boxes.cls)
    for box, cls, tid in zip(result.boxes.xyxy, result.boxes.cls, ids):
        x1,y1,x2,y2 = map(int, box)
        cx, cy = (x1+x2)/2, (y1+y2)/2

        # Kalman
        if tid not in kalman_filters:
            kf = create_kalman()
            kf.statePost = np.array([[cx],[cy],[0],[0]], np.float32)
            kalman_filters[tid] = kf
        else:
            kf = kalman_filters[tid]

        px,py,pvx,pvy = kf.predict().flatten()
        kf.correct(np.array([[cx],[cy]], np.float32))

        # draw
        label = f"{model.names[int(cls)]} ID:{int(tid)}"
        cv2.rectangle(annotated,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(annotated,label,(x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
        cv2.circle(annotated,(int(px),int(py)),4,(0,0,255),-1)
        cv2.arrowedLine(annotated,
                        (int(cx),int(cy)),
                        (int(px),int(py)),
                        (0,0,255),2,tipLength=0.3)

        # buffer for narration
        with buffer_lock:
            scene_buffer.append({
                "track_id": tid,
                "class":    model.names[int(cls)],
                "cx":       cx,
                "cy":       cy,
                "vx":       pvx,
                "vy":       pvy
            })

    # send annotated frame back
    _, buf = cv2.imencode(".jpg", annotated)
    jpg_b64 = base64.b64encode(buf).decode("utf-8")
    emit("annotated_frame", {"data": f"data:image/jpeg;base64,{jpg_b64}"})

# ─── START THE GPT BACKGROUND TASK & SERVER ────────────────────────────────────
if __name__ == "__main__":
    # start narration thread
    threading.Thread(target=send_to_gpt, daemon=True).start()
    # run flask
    socketio.run(app, host="0.0.0.0", port=5000)
