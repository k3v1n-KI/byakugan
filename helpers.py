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
# from flask import Flask, render_template
# from flask_socketio import SocketIO, emit
# from ultralytics import YOLO
# import supervision as sv


# ─── CONFIG ────────────────────────────────────────────────────────────────────
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
            os.remove(tmp_file)
    return uri