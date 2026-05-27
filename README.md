# KAI - Automated Energy Management System

> _An automated infrastructure management system focused on reducing power usage. Hooks into pre-existing infrastructure to control appliances when not in use. Designed to be extremely flexible, low maintenance, and scalable as needed._

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![MQTT](https://img.shields.io/badge/MQTT-Mosquitto-orange.svg)](https://mosquitto.org/)
[![YOLOv11](https://img.shields.io/badge/YOLOv11-Computer%20Vision-green.svg)](https://github.com/ultralytics/ultralytics)
[![Flutter](https://img.shields.io/badge/UI-Flutter-blue.svg)](https://flutter.dev/)

---

## Overview

KAI is an IoT system designed for large organizations to save energy by automatically turning off appliances when rooms are unoccupied. Using existing camera feeds and YOLOv11 computer vision, the system detects human presence and intelligently manages power to connected devices. Refer to the system overview image or the system architecture to know more about how it works.

### Key Features

**Automated Power Management** - Turns off appliances after detecting 300s (configurable) of no human presence.
**Computer Vision** - YOLOv11-based person detection from camera feeds  
**Near Real-time Dashboard** - Flutter UI (Android/Windows) showing live camera status and manual controls.
**MQTT Architecture** - Scalable, real-time messaging with Eclipse Mosquitto  
**One-Command Launch** - Start entire system with `python start.py`

---

The system is designed around a decoupled, event-driven pattern mediated by the **Mosquitto MQTT Broker** for low-latency control flow, combined with a **Shared File Cache (`images_ready/`)** and an **HTTP API Server** for high-bandwidth data retrieval (like camera feeds, configs, and YOLO bounding box outputs). 

### Architecture Diagram

```
                                 ┌───────────────────────┐
                                 │     Image Server      │
                                 │  (YOLOv11 Detector)   │
                                 └────▲─────────────┬────┘
                                      │             │
                   kai/force_request  │             │ Writes processed images
                   (Subscribe Feed)   │             │ and bounding boxes (.json)
                                      │             ▼
               ┌──────────────────────┴─────────────┴───────────────┐
               │               MQTT Broker (Mosquitto)              │
               └──────▲─────────────▼▲──────────────▼▲───────────┬──┘
                      │             ││              ││           │
    kai/force_request │             ││              ││           │ kai/power
    (Publish)         │  kai/force_ ││ kai/control  ││ kai/power │ (Subscribe)
                      │  served     ││ (Publish)    ││ (Publish) │
                      │  (Subscribe)││              ││           │
                      ▼             │▼              │▼           ▼
               ┌──────┴─────────────┴┐ ┌────────────┴┴┐ ┌────────┴─────┐
               │      Dashboard      │ │   Control    │ │  Power Feed  │
               │    (Flutter UI)     │ │   Server     │ │ (Appliances) │
               └──────────▲──────────┘ └──────┬───────┘ └──────────────┘
                          │                   │
                          │                   │ Reads camera occupancy
                          │                   │ status directly from disk
                          │                   ▼
                          │         ┌─────────────────────────────────┐
                          │         │          images_ready/          │
                          │         │  (Shared Directory / Data Cache)│
                          │         └────────────────▲────────────────┘
                          │                          │
                          │                          │ Reads images and
                          │ HTTP REST Requests       │ bounding box metadata
                          │ (Port 8000)              │
                          ▼                          │
               ┌─────────────────────────────────────┴────────────────┐
               │                   HTTP API Server                    │
               │       Serves camera metadata and static images       │
               └──────────────────────────────────────────────────────┘
```

### Component Roles & Feeds

1. **Image Server (YOLOv11)**
   * Captures and analyzes static frames from cameras to detect human presence.
   * **Subscribes to** `kai/force_request` to run immediate YOLO detection on a requested camera.
   * **Publishes to** `kai/force_served` once a force-requested check is processed and tagged in the filesystem.

2. **Dashboard (Flutter UI)**
   * Displays room occupancy status (indicated by Green/Red indicators based on YOLO image tags).
   * **Fetches** camera list and streams the actual camera images over **HTTP** from the HTTP API Server.
   * **Publishes to** `kai/force_request` when the user manually requests an immediate room check.
   * **Subscribes to** `kai/force_served` to know exactly when to pull the updated camera feed/image.
   * **Publishes to** `kai/control` for manual power overrides (`SET_CAM_<id>_ON` or `OFF`).

3. **Control Server**
   * Automatically monitors camera presence statuses from the tagged images directory.
   * **Subscribes to** `kai/control` to receive manual power overrides from the Dashboard.
   * **Publishes to** `kai/power` to toggle appliances (either via automated 10-consecutive "NO" detection timeout or manual overrides).

4. **Power Feed (Appliances / Relays)**
   * **Subscribes to** `kai/power` to toggle connected physical relays and appliances.

5. **HTTP API Server (Port 8000)**
   * Built on a lightweight HTTP protocol to decouple static resource retrieval from the real-time MQTT message bus.
   * **Serves `/api/cameras`**: Exposes the discovery metadata, YOLO detection state, and bounding box coordinates for each camera.
   * **Serves `/images/*`**: Streams the tagged JPEG images from the `images_ready/` directory directly to the UI.
   * **Serves `/api/config`**: Allows reading and updating runtime configurations.

---

## Quick Start

### Prerequisites

- **Python 3.8+**
- **Mosquitto MQTT Broker**
- **Camera images** in `images_src/` folder (format: `CAM_*.jpg`)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Cosmic-Infinity/kai.git
cd kai

# 2. Install Mosquitto MQTT broker
choco install mosquitto  # Windows with Chocolatey
# OR download from: https://mosquitto.org/download/

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Secure Mosquitto & Start Service
# Run the interactive setup tool in an Administrator PowerShell (Windows):
python tools/setup_mqtt_auth.py
```

### Run the System

**One-Command Launch (Recommended):**

```bash
python start.py
```

This launches all modules in separate windows:

- Image Server (processes camera feeds)
- Control Server (manages power)
- Dashboard UI (Flutter Desktop/Mobile via API)
- MQTT Monitor (traffic viewer)

**Manual Launch:**

```bash
# Terminal 1 - Image Server
python modules/image_server.py

# Terminal 2 - Control Server
python modules/control_server.py

# Terminal 3 - Dashboard (Flutter)
cd dashboard
flutter run

# Terminal 4 - Monitor (optional)
python tools/monitor_mqtt.py
```

### Test MQTT Connection

```bash
python tools/test_mqtt.py
# Should show: [OK] Successfully connected to MQTT broker!
```

---

## Project Structure

```text
kai/
├── .github/                 # GitHub actions configurations
│   └── workflows/
│       ├── flutter_build_android.yml # Android APK compilation workflow
│       └── flutter_build_windows.yml # Windows App compilation workflow
├── docs/                    # Technical specifications & documentation
│   └── system.txt
├── models/                  # Object detection weights
│   ├── yolo11m.pt
│   └── yolo11s.pt
├── modules/                 # Application core modules
│   ├── feeds.py             # MQTT feed management
│   ├── image_server.py      # Camera processing and YOLO detector
│   ├── control_server.py    # Automatic appliance power control
│   └── http_api.py          # Frame transmission web server
├── dashboard/               # Flutter Source code (UI for Android/Windows)
├── tools/                   # Debugging and utility scripts
│   ├── setup_mqtt_auth.py   # Interactive Mosquitto security setup
│   ├── test_mqtt.py         # Test broker connection
│   ├── monitor_mqtt.py      # Print real-time MQTT message feeds
│   └── finetune.py          # Fine-tuning utilities
├── mqtt_config.py           # Main broker configurations
├── requirements.txt         # Python dependencies
├── README.md                # System documentation
├── start.py                 # One-command system environment launcher
├── images_src/              # Mock raw input frames
└── images_ready/            # Processed YOLO results cache
```

---

## Configuration

### MQTT Broker (`mqtt_config.py`)

```python
# Broker settings
MQTT_BROKER_HOST = "127.0.0.1"  # IP of your broker (use 127.0.0.1 for local)
MQTT_BROKER_PORT = 1883         # Default MQTT port

# Authentication (as required)
MQTT_USERNAME = None            # Set as required
MQTT_PASSWORD = None            # Set as required

# Security (deployment)
MQTT_USE_TLS = False            # Enable for encrypted connections
MQTT_QOS = 1                    # Quality of Service (0, 1, or 2)
```

### Image Server

- **Capture Interval**: 60 seconds
- **Batch Processing**: 25 images at a time
- **Model**: YOLOv11 (auto-detects fine-tuned version)

### Control Server

- **Control Response**: ~100ms (instant)
- **Status Monitoring**: Every 30 seconds
- **Auto Power-Off**: 10 consecutive "NO" detections

### Dashboard

- **Refresh Interval**: 30 seconds
- **Visual Indicators**: Green (person detected), Red (no person)
- **Controls**: Force update, Power toggle per camera

---

## How It Works

### 1. Image Processing

- Reads images from `images_src/` every 60 seconds
- Runs YOLOv11 person detection
- Tags images as `CAM_name_YES.jpg` (person) or `CAM_name_NO.jpg` (empty)
- Saves to `images_ready/`
- Handles force update requests instantly

### 2. Dashboard (UI)

- Displays all camera feeds with status borders
- **Green border**: Person detected
- **Red border**: No person detected
- **Force Update**: Request immediate camera check
- **Toggle Power**: Manual appliance control

### 3. Control System

- **Fast Thread**: Checks UI commands every 100ms
- **Slow Thread**: Monitors camera status every 30s
- Auto power-off after 10 consecutive "NO" readings
- Writes power commands to POWER feed

### 4. MQTT Communication

- **kai/force_request**: Dashboard → Image Server
- **kai/force_served**: Image Server → Dashboard
- **kai/control**: Dashboard → Control Server
- **kai/power**: Control Server → Appliances

---

## Testing

### Test MQTT Connection

```bash
python tools/test_mqtt.py
```

### Monitor Real-Time Traffic

```bash
python tools/monitor_mqtt.py
```

Shows color-coded messages:

- **Force requests** (yellow)
- **Force served** (green)
- **Control commands** (blue)
- **Power commands** (red)

### Test Individual Modules

```bash
# Test image processing
python modules/image_server.py

# Test control logic
python modules/control_server.py

# Test UI
cd dashboard
flutter run
```

---

## Troubleshooting

| Problem                   | Solution                                  |
| ------------------------- | ----------------------------------------- |
| **"Connection refused"**  | Start Mosquitto: `net start mosquitto`    |
| **"Import paho.mqtt"**    | Install: `pip install paho-mqtt`          |
| **Images not processing** | Check `images_src/` has `CAM_*.jpg` files |
| **Messages not received** | Verify topic names in `mqtt_config.py`    |

---

## Built With

- **[Python](https://www.python.org/)** - Core language
- **[YOLOv11](https://github.com/ultralytics/ultralytics)** - Computer vision
- **[Eclipse Mosquitto](https://mosquitto.org/)** - MQTT broker
- **[Paho MQTT](https://www.eclipse.org/paho/)** - MQTT client
- **[Flutter](https://flutter.dev/)** - Dashboard UI
- **[PyTorch](https://pytorch.org/)** - ML framework

---

## License

See [LICENSE](LICENSE) file for details.

---

**Parts of this description was generated by AI. Please keep an eye out for inconsistencies.**
