# Byakugan

**Byakugan** is a real-time AI-powered navigation assistant designed to help visually impaired users better understand and safely navigate their surroundings. The system combines computer vision, object tracking, temporal context modeling, and neural language generation to transform live camera input into contextual spoken navigation guidance.

The project demonstrates how modern AI systems — including object detection models, temporal reasoning, and language models — can be integrated into a practical assistive technology pipeline.

---

## Overview

Byakugan captures live camera frames from a mobile device, analyzes the environment using computer vision models, builds a structured representation of the scene, and generates natural-language navigation guidance that is delivered to the user through audio.

The system integrates multiple AI components:

- **Object Detection** using YOLOv8-World
- **Multi-object Tracking** using ByteTrack
- **Trajectory Estimation** using Kalman Filters
- **Scene Understanding**
- **Temporal Context Modeling**
- **Natural Language Navigation Guidance**
- **Text-to-Speech Audio Feedback**

This architecture allows the system to provide **real-time situational awareness** to users navigating dynamic environments.

---

## System Architecture

The system is composed of two primary components:

### Mobile Application (Flutter)

The mobile application is responsible for:

- Capturing live camera frames
- Sending frames to the server for processing
- Receiving navigation responses
- Converting responses to audio feedback for the user

Main modules include:

- **Camera Module** – Captures frames from the device's rear camera
- **Request Handler** – Sends frames to the backend server at regular intervals
- **Response Parser** – Interprets server responses
- **TTS Module** – Converts navigation instructions into speech
- **State Management** – Maintains application state and handles asynchronous responses

---

### Backend Server (Flask)

The backend server performs the core AI processing pipeline.

Incoming frames are processed through the following stages:

1. **Object Detection**
2. **Scene Construction**
3. **Temporal Context Integration**
4. **Navigation Reasoning**
5. **Response Generation**

---

## Image Processing Pipeline

### Object Detection

The system uses **YOLOv8-World** to detect objects within each incoming frame.

This model performs real-time detection of objects such as:

- pedestrians
- vehicles
- bicycles
- obstacles
- environmental structures

The model outputs:

- bounding box coordinates
- object class labels
- confidence scores

These detections form the foundation for scene interpretation.

---

### Multi-Object Tracking

Detected objects are tracked across frames using **ByteTrack**, which assigns persistent IDs to objects over time.

Tracking enables the system to:

- maintain object identity
- detect approaching or departing objects
- estimate motion patterns

This allows the system to reason about dynamic environments rather than individual static frames.

---

### Trajectory Estimation

Each tracked object is associated with a **Kalman Filter**, which estimates:

- object position
- object velocity
- predicted future location

Kalman filtering reduces noise in detection results and enables **trajectory prediction**, allowing the system to identify potential hazards before they occur.

---

## Temporal Context Manager

To improve the quality and consistency of navigation instructions, the system includes a **Temporal Context Manager**.

Rather than treating each frame independently, the system maintains a short-term memory of recent navigation events.

The context manager stores:

- previous scene descriptions
- previously issued navigation guidance
- key detected obstacles
- the number of critical hazards detected

This historical context is injected into the language model prompt, allowing the system to generate **more coherent and situationally aware guidance**.

Example:

Instead of:

> "There is a person ahead."

The system may generate:

> "The person detected earlier is now approaching closer from the right."

The Temporal Context Manager also performs simple trend analysis to identify:

- recurring obstacles
- increasing or decreasing danger levels

---

## Navigation Guidance Generation

Scene information and temporal context are passed to a language model to generate natural-language navigation instructions.

The model converts structured scene data into concise guidance such as:

- describing nearby objects
- warning about approaching hazards
- suggesting directional adjustments

The resulting instructions are designed to be **clear, brief, and immediately actionable**.

---

## Text-to-Speech Feedback

Navigation guidance is converted into speech using a text-to-speech system and delivered to the mobile device.

This enables users to receive real-time auditory feedback while navigating their environment.

---

## Technologies Used

### Computer Vision
- YOLOv8-World
- OpenCV
- Ultralytics

### Tracking & Motion Estimation
- ByteTrack
- Kalman Filters

### Backend
- Python
- Flask

### Mobile Development
- Flutter
- Dart

### AI / Language Models
- GPT-based reasoning (development stage)
- Local LLM experimentation (planned)

---

## Key Features

- Real-time object detection
- Multi-object tracking
- Motion trajectory estimation
- Temporal scene context
- Natural-language navigation instructions
- Audio-based feedback
- Mobile-first assistive interface

---

## Future Work

Planned improvements include:

- More robust object detection models trained for assistive navigation
- Local LLM integration to reduce latency and API dependency
- Real-time streaming communication (WebSockets)
- Enhanced trajectory prediction and hazard forecasting
- Improved accessibility features in the mobile interface

---

## Research Motivation

Byakugan explores how modern AI systems can be integrated to improve **human-centered safety applications**.

The project investigates challenges related to:

- grounding language models in real-world sensor data
- reducing hallucination in AI-generated instructions
- maintaining contextual consistency across time
- building reliable AI systems for safety-critical environments

---

## Contributors

- **Kevin Igweh**
- **Jikesh Thapa**

Algoma University  
Neural Networks & Deep Learning

---

## License

This project is for academic and research purposes.
