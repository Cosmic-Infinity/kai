# KAI - Automated Energy Management System

> _An automated infrastructure management system focused on reducing power usage. Hooks into pre-existing infrastructure to control appliances when not in use. Designed to be extremely flexible, low maintenance, and scalable as needed._

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![MQTT](https://img.shields.io/badge/MQTT-Mosquitto-orange.svg)](https://mosquitto.org/)
[![YOLOv11](https://img.shields.io/badge/YOLOv11-Computer%20Vision-green.svg)](https://github.com/ultralytics/ultralytics)
[![Kivy](https://img.shields.io/badge/UI-Kivy-purple.svg)](https://kivy.org/)

---

## 🎯 Overview

KAI is an IoT system designed for large organizations to save energy by automatically turning off appliances when rooms are unoccupied. Using camera feeds and YOLOv11 computer vision, the system detects human presence and intelligently manages power to connected devices. Refer to the system overview image or the system architecture to know more about how it works.

### Key Features

✨ **Automated Power Management** - Turns off appliances after detecting 300s (configurable) of no human presence.
🎥 **Computer Vision** - YOLOv11-based person detection from camera feeds  
📊 **Real-time Dashboard** - Kivy UI showing live camera status and manual controls.
🔌 **MQTT Architecture** - Scalable, real-time messaging with Eclipse Mosquitto  
🚀 **One-Command Launch** - Start entire system with `python start.py`

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      MQTT Broker (Mosquitto)                │
└──────────────┬──────────────────┬────────────────┬──────────┘
               │                  │                │
       ┌───────▼────────┐  ┌─────▼──────┐  ┌──────▼────────┐
       │ Image Server   │  │  Control    │  │   Dashboard   │
       │ (YOLOv11)      │  │  Server     │  │   (Kivy UI)   │
       │                │  │             │  │               │
       │ • Detects      │  │ • Monitors  │  │ • Shows feeds │
       │   persons      │  │   status    │  │ • Force update│
       │ • Tags images  │  │ • Auto      │  │ • Manual      │
       │ • Force update │  │   power-off │  │   control     │
       └────────────────┘  └─────────────┘  └───────────────┘
               │                  │                │
               └──────────────────┼────────────────┘
                                  │
                           ┌──────▼────────┐
                           │  Power Feed   │
                           │  (Appliance   │
                           │   Control)    │
                           └───────────────┘
```

---

## 🚀 Quick Start

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

# 4. Start Mosquitto service
net start mosquitto  # Windows
# sudo systemctl start mosquitto  # Linux
```

### Run the System

**One-Command Launch (Recommended):**

```bash
python start.py
```

This launches all modules in separate windows:

- Image Server (processes camera feeds)
- Control Server (manages power)
- Dashboard UI (Kivy interface)
- MQTT Monitor (traffic viewer)

**Manual Launch:**

```bash
# Terminal 1 - Image Server
python modules/image_server.py

# Terminal 2 - Control Server
python modules/control_server.py

# Terminal 3 - Dashboard
python modules/ui.py

# Terminal 4 - Monitor (optional)
python monitor_mqtt.py
```

### Test MQTT Connection

```bash
python test_mqtt.py
# Should show: ✓ Successfully connected to MQTT broker!
```

---

## 📁 Project Structure

```
kai/
├── start.py                 # ⭐ One-command launcher
├── test_mqtt.py             # Test MQTT connection
├── monitor_mqtt.py          # Monitor MQTT traffic
├── mqtt_config.py           # MQTT configuration
├── feeds.py                 # MQTT feed system
├── requirements.txt         # Dependencies
├── README.md                # This file
├── system.txt               # System specification
│
├── modules/                 # Application modules
│   ├── image_server.py      # Camera processing
│   ├── control_server.py    # Power control
│   └── ui.py                # Dashboard UI
│
├── images_src/              # Source camera images
├── images_ready/            # Processed images (status tagged)
├── feeds/                   # Legacy file-based feeds
└── human-detection-in-classroom/  # Training dataset
```

---

## 🔧 Configuration

### MQTT Broker (`mqtt_config.py`)

```python
# Broker settings
MQTT_BROKER_HOST = "localhost"  # Change for remote broker
MQTT_BROKER_PORT = 1883         # Default MQTT port

# Authentication (optional)
MQTT_USERNAME = None            # Set if required
MQTT_PASSWORD = None            # Set if required

# Security (production)
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

## 📊 How It Works

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

## 🧪 Testing

### Test MQTT Connection

```bash
python test_mqtt.py
```

### Monitor Real-Time Traffic

```bash
python monitor_mqtt.py
```

Shows color-coded messages:

- 📤 **Force requests** (yellow)
- ✓ **Force served** (green)
- 🎮 **Control commands** (blue)
- ⚡ **Power commands** (red)

### Test Individual Modules

```bash
# Test image processing
python modules/image_server.py

# Test control logic
python modules/control_server.py

# Test UI
python modules/ui.py
```

---

## 🐛 Troubleshooting

| Problem                   | Solution                                  |
| ------------------------- | ----------------------------------------- |
| **"Connection refused"**  | Start Mosquitto: `net start mosquitto`    |
| **"Import paho.mqtt"**    | Install: `pip install paho-mqtt`          |
| **Slow UI response**      | Already fixed! Now ~100ms response time   |
| **Images not processing** | Check `images_src/` has `CAM_*.jpg` files |
| **Messages not received** | Verify topic names in `mqtt_config.py`    |

---

## 🛠️ Built With

- **[Python](https://www.python.org/)** - Core language
- **[YOLOv11](https://github.com/ultralytics/ultralytics)** - Computer vision
- **[Eclipse Mosquitto](https://mosquitto.org/)** - MQTT broker
- **[Paho MQTT](https://www.eclipse.org/paho/)** - MQTT client
- **[Kivy](https://kivy.org/)** - Dashboard UI
- **[PyTorch](https://pytorch.org/)** - ML framework

---

## 📄 License

See [LICENSE](LICENSE) file for details.

---

**Parts of this description was generated by AI. Please keep an eye out for inconsistencies.**
