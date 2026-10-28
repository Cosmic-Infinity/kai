# kai

> _An automated infrastructure management system focused on reducing energy consumption. Hooks into pre-existing infrastructure to control appliances when not in use. Designed to be extremely flexible, low maintenance, and scalable as needed._

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![MQTT Server](https://img.shields.io/badge/MQTT_Server-Mosquitto-orange.svg)](https://mosquitto.org/)
[![MQTT Client](https://img.shields.io/badge/MQTT_Client-Paho_MQTT-8E24AA.svg)](https://www.eclipse.org/paho/)
[![Computer Vision](https://img.shields.io/badge/Computer_Vision-YOLOv11-green.svg)](https://github.com/ultralytics/ultralytics)
[![ML](https://img.shields.io/badge/ML-PyTorch-EE4C2C.svg)](https://pytorch.org/)
[![UI](https://img.shields.io/badge/UI-Flutter-blue.svg)](https://flutter.dev/)

---

## Overview

kai is an automation system designed for large organizations to save energy by automatically turning off appliances when rooms are unoccupied. Using pre-existing camera and network infrastructure, and computer vision, the system detects human presence and intelligently manages power to connected devices.

It is designed around a decoupled, event-driven pattern mediated by a Mosquitto MQTT Broker for low-latency control flow, combined with an HTTP API Server for high-bandwidth data retrieval (say, camera feeds, configs, etc). Everything runs locally in the organization's network and hardware for absolute privacy and control. Hence each deployment is self-contained, and can have modular services attached for easy extensibility.

It uses timers at multiple levels to monitor and control appliances, but can be triggered explicitly from the dashboard for manual checks or to override the system. It is designed to be self-healing and fault tolerant, while maintaining strict access controls both internally, and externally.

A thing to note, the system automatically only ever writes `OFF` signals to turn off appliances. The power `ON` signal is always a manual override. Hence **Auto power OFF** and **Manual power ON**. This is a deliberate design choice.


### Core Architecture

![kai Architecture Diagram](images_screenshots/kai_architecture.png)

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
   * **Secure Storage**: Persists authentication keys securely using OS-level encrypted vaults (Windows Credential Manager / Android Keystore).

3. **Control Server**
   * Automatically monitors camera presence statuses from the tagged images directory.
   * **Subscribes to** `kai/control` to receive manual power overrides from the Dashboard.
   * **Publishes to** `kai/power` to toggle appliances (either via automated timeout after 10 consecutive "NO" checks or manual overrides).

4. **Power Feed ( `kai/power` )**  This is the system's output, where Appliances/Relays subscribes to, to know when to toggle power.

5. **HTTP API Server (Port 8000)**
   * Built on a lightweight HTTP protocol to decouple static resource retrieval from the real-time MQTT message bus.
   * **Serves `/api/cameras`**: Exposes the discovery metadata, YOLO detection state, bounding box coordinates, and last known `power_state` for each camera.
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
choco install mosquitto          # Windows with Chocolatey
sudo apt install mosquitto mosquitto-clients   # Debian/Ubuntu
brew install mosquitto           # macOS with Homebrew
# OR download from: https://mosquitto.org/download/

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Secure Mosquitto & Start Service
# Run the interactive setup tool (on Windows it asks for UAC elevation itself;
# on Linux/macOS run it with sudo and configure Mosquitto manually):
python configure_kai.py
```

### Run the System

**One-Command Launch (Recommended):**

```bash
python start.py
```

This launches the backend and monitoring modules in separate terminal windows (PowerShell on Windows, Terminal on macOS, gnome-terminal / konsole / xfce4-terminal / xterm on Linux). On a headless machine with no terminal available, the modules run in the background and write their output to `logs/<module>.log` instead:

- Image Server (processes camera feeds)
- Control Server (manages power)
- HTTP API Server (exposes REST endpoints and video frames)
- MQTT Monitor (traffic viewer)

> [!IMPORTANT]
> The Flutter Dashboard UI is not launched automatically by `start.py`. You must run it separately in a new terminal:
> ```bash
> cd dashboard
> flutter run
> ```

**Manual Launch:**

```bash
# Terminal 1 - Image Server
python modules/image_server.py

# Terminal 2 - Control Server
python modules/control_server.py

# Terminal 3 - HTTP API Server
python modules/http_api.py

# Terminal 4 - Dashboard (Flutter)
cd dashboard
flutter run

# Terminal 5 - Monitor (optional)
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
├── config/                  # Centralized system configurations
│   └── config.json          # Single Source of Truth configuration file (ships with SAMPLE credentials!)
├── dashboard/               # Flutter Source code (UI for Android/Windows)
├── docs/                    # Technical specifications & documentation
│   └── system.txt
├── images_ready/            # Processed YOLO results cache
├── images_screenshots/      # System architecture & dashboard screenshots
├── images_src/              # Mock raw input frames
├── models/                  # Object detection weights
│   ├── yolo11m.pt
│   └── yolo11s.pt
├── modules/                 # Application core modules
│   ├── config_manager.py    # Configuration loader and namespace manager
│   ├── control_server.py    # Automatic appliance power control
│   ├── feeds.py             # MQTT feed management
│   ├── http_api.py          # Frame transmission web server
│   ├── image_server.py      # Camera processing and YOLO detector
│   └── logger.py            # Unified colorful logging module
├── state/                   # Runtime state (git-ignored), e.g. last known power state per camera
├── logs/                    # Module output when started without a terminal (git-ignored)
├── tools/                   # Debugging and utility scripts
│   ├── finetune.py          # Fine-tuning utilities
│   ├── monitor_mqtt.py      # Print real-time MQTT message feeds
│   ├── security_stresstest.py # Run automated security audit and stress tests
│   └── test_mqtt.py         # Test broker connection
├── configure_kai.py         # Interactive wizard script for Mosquitto & security setup
├── README.md                # System documentation
├── requirements.txt         # Python dependencies
└── start.py                 # One-command system environment launcher
```

---

## Configuration

### MQTT Broker & Access Control (ACL)

The system automatically configures Mosquitto with granular, multi-user Access Control Lists (ACL) to strictly isolate system component privileges. During configuration, internal server accounts are provisioned with secure, randomly-generated credentials, while the dashboard username/password is set up interactively:

*   **Dashboard User (`kai_dashboard` or custom)**: Used by the Flutter app. Has restricted `read` access for telemetry (`kai/power`, `kai/force_served`) and restricted `write` access for user actions (`kai/control`, `kai/force_request`).
*   **Image Server User (`kai_image_server`)**: Has restricted `read` access for `kai/force_request` and `write` access for `kai/force_served`.
*   **Control Server User (`kai_control_server`)**: Has restricted `read` access for `kai/control` and `write` access for `kai/power`.
*   **Edge Device User (`kai_edge_device` or custom)**: Used by appliances/relays. Has restricted `read` access for `kai/power` only.
*   **Traffic Monitor User (`kai_monitor`)**: Full read-only access (`kai/#`) to safely trace and log system events.
*   **Developer Mode Bypass**: Optionally enables a single shared developer credential (`kai_admin` / `kai_developer`) across all modules.

To set up production credentials and generate these secure boundaries automatically, run:
```bash
python configure_kai.py
```

> [!WARNING]
> **The `config/config.json` committed to this repository contains SAMPLE credentials only** (dashboard login, edge device login, internal server passwords and the HTTP API key) so the application can be showcased. They are public and must never be used for a real deployment. Run `python configure_kai.py` to replace them: the internal server passwords are regenerated as random 128-bit values (`secrets.token_hex(16)`), and you choose your own dashboard / edge device passwords and API key. While the sample values are still in place, the config carries a `_NOTICE` entry and `start.py` prints a warning; the wizard removes the notice once real credentials are set.

```python
# Broker settings (config/config.json)
MQTT_BROKER_HOST = "127.0.0.1"
MQTT_BROKER_PORT = 1883
```

## System Components & Workflows

### 1. Image Server (YOLOv11 Detector)
- **Image Processing**: Reads raw frames from `images_src/` every 60 seconds (default), runs YOLOv11 person detection, tags output as `CAM_<name>_YES.jpg` (person present) or `CAM_<name>_NO.jpg` (empty), and caches it in `images_ready/`.
- **Force Updates**: Instantly captures and processes targeted camera requests on-demand.
- **Fail-safe on errors**: If a frame cannot be processed (unreadable image, model failed to load, prediction error) the error is printed and that camera's previous status is kept. A failed detection is never treated as an empty room, because `NO` eventually powers appliances off. For a force request that fails, no `force_served` message is sent.
- **Model**: YOLOv11 (auto-detects fine-tuned model weights).

### 2. Dashboard (Flutter UI)
- **Telemetry Display**: Reads camera occupancy statuses via the HTTP API Server, updating cards with high-contrast color borders: **Green** (person detected) or **Red** (room empty).
- **Control Actions**: Publishes manual appliance override commands (`kai/control`) and on-demand room check requests (`kai/force_request`) instantly.

### 3. Control Server
- **Dual-Thread Engine**:
  - **Fast Thread (100ms)**: Regularly checks and executes manual dashboard power override commands instantly.
  - **Slow Thread (30s)**: Regularly evaluates camera occupancy logs. Auto-powers off appliances after 10 consecutive checks that find the room "NO" (empty) to prevent energy waste. See [Timing Model](#timing-model) below.
- **Actuator Feed**: Publishes final power state decisions directly to the `kai/power` feed.
- **Power State Record**: Every published `ON`/`OFF` is also saved to `state/power_state.json`. The HTTP API returns it as `power_state` in `/api/cameras`, so a freshly started dashboard shows the true state on its first fetch instead of assuming `ON` (after that, live `kai/power` messages keep it current).

### Timing Model

The system's timers are **deliberately asynchronous**: each component runs on its own clock and nothing waits for anything else.

| Timer | Default | What it drives |
| :--- | :--- | :--- |
| `IMAGE_SERVER_INTERVAL` | 60 s | How often the Image Server re-analyses every camera frame |
| `CONTROL_SERVER_INTERVAL` | 30 s | How often the Control Server reads the current status of every camera |
| `INACTIVITY_THRESHOLD` | 10 | Consecutive Control Server checks that must find a room empty before auto power-off |
| `DASHBOARD_INTERVAL` | 30 s | How often the Dashboard re-fetches cameras (force requests and MQTT events update it in between) |

- **Inactivity counts checks, not frames.** Every control check reads whatever status is current, so one frame can be seen by several checks. The auto-off delay is therefore about `INACTIVITY_THRESHOLD × CONTROL_SERVER_INTERVAL` (5 minutes by default), whatever the capture interval. A single "YES" resets the count.
- **Frame timestamps.** For every processed frame the Image Server records, in `state/frames.json`:
  - `captured_at`: the camera's EXIF capture time if present, otherwise the file's modification time
  - `processed_at`: when it was analysed, by the server's clock
  - `camera_clock_offset`: how far the camera's EXIF clock disagrees with the file time

  The HTTP API returns all three in `/api/cameras`.
- **Camera clocks are not trusted.** Cameras don't always have the right time, so `captured_at` is informational only. Every timing decision uses `processed_at`, which comes from the server's clock. A camera whose clock is more than 5 minutes off gets a one-time warning in the Image Server log.
- **Stale frames are not evidence.** If a camera's frame hasn't been processed for `3 × IMAGE_SERVER_INTERVAL` (for example, the Image Server stopped), the Control Server logs a warning and holds that camera's inactivity count. It doesn't add to the count or reset it, and it takes no automatic action until fresh frames arrive. This works the same way whether the last status was "YES" or "NO".

### 4. MQTT Communication & Feeds
The system is built on a decoupled, event-driven pattern using isolated MQTT topics:
- **`kai/force_request`**: Dashboard → Image Server (requests an immediate YOLO check)
- **`kai/force_served`**: Image Server → Dashboard (signals that a forced update is complete)
- **`kai/control`**: Dashboard → Control Server (triggers manual appliance overrides)
- **`kai/power`**: Control Server → Appliances (actuates physical appliance power changes)

### 5. HTTP API Server (Gateway)
- **Video Frame Streaming**: Securely serves captured and tagged presence frames (`images_ready/`) to the Flutter Dashboard.
- **Access Gateway**: Enforces secure token-based (Bearer Token) authentication to restrict frame viewing and API access to validated clients.
- **Config & Path Masking**: Shields system config files (`config/config.json`) and internal server directories from configuration exposures.

### 6. Interactive Setup Wizard & Diagnostics (`configure_kai.py`)
- **Interactive Secure Setup**: Automates generation of cryptographically secure internal credentials, provisions multi-client Mosquitto password databases, and configures granular ACL policies.
- **Targeted Credentials Reset**: Allows users to dynamically update individual module keys (such as dashboard or edge credentials) while keeping the rest of the production architecture isolated.
- **Live System Diagnostics**: Runs complete loopback diagnostics checking connection status and authorization access profiles for all five system roles, testing the HTTP API gateway integrity, and displaying a high-contrast console status card.

### 7. Security Stress Testing (`tools/security_stresstest.py`)
- **Automated Penetration Auditor**: A dynamic test script evaluating all system safety features. 
- **Security Checkpoints**: Automatically audits HTTP auth bypass attempts, malformed payload injections, MQTT credential gates, ACL role boundary isolation, and invalid wildcard topic subscriptions.

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

### Automated Security & Privilege Auditing

```bash
python tools/security_stresstest.py        # asks for confirmation; pass --yes to skip
```

> [!CAUTION]
> The audit **stops whatever is listening on port 8000** (your running HTTP API) so it can test a fresh instance, and relaunches the API when it finishes (in a new window on Windows, as a background process elsewhere). Dashboards lose their connection in the meantime; close your old HTTP API window afterwards and restart any other service that used the port. The audit also temporarily changes `DASHBOARD_INTERVAL` in `config/config.json`, and restores `config.json` to its original contents when it ends (even if interrupted).

Runs a portable diagnostic test suite confirming strict privilege isolation limits:
* **HTTP API Sanitization**: Verifies that anonymous queries are blocked, write calls are whitelisted/bounds-checked, and GET config payloads mask all administrative credentials and connection properties.
* **MQTT Password Gatekeeper**: Stress-tests authentication, confirming that dictionary attacks or default logins are strictly rejected.
* **Granular ACL Enforcement**: Simulates Dashboard role bypasses (unauthorized writes to `kai/power` or reads from `kai/control`) and wildcard sniffing attempts (`kai/#`) to confirm they are securely dropped and filtered.

---

## Troubleshooting

The **kai** system includes built-in diagnostics to isolate problems quickly. If you run into issues, try executing the system loopback checks first:
```bash
python configure_kai.py  # Choose option 4 to run system diagnostics
```

Refer to the table below for common symptoms, root causes, and verified fixes:

| Symptom / Error | Potential Cause | Verified Resolution |
| :--- | :--- | :--- |
| **MQTT `Connection refused`** <br>`[WinError 10061]` | Mosquitto Broker service is offline, or connection credentials are out of sync. | 1. Ensure the Mosquitto service is running: `net start mosquitto` (Windows Admin PowerShell), `sudo systemctl start mosquitto` (Linux) or `brew services start mosquitto` (macOS). <br>2. Run `python configure_kai.py` to regenerate and verify credentials in `config/config.json`. |
| **Mosquitto Exit Code `13`** <br>`(Permission Denied)` | Mosquitto parser failed to read configuration lines containing double quotes or backslashes in paths (e.g. `C:\Program Files`). | Do **NOT** use double quotes around paths in `mosquitto.conf`. Run `python configure_kai.py` to rewrite the `password_file` / `acl_file` entries as clean, unquoted forward-slash paths (e.g., `C:/Program Files/mosquitto/passwd`). |
| **Constant MQTT Disconnects** <br>or telemetry dropping | Overlapping duplicate Client IDs connected to the same broker, or client ACL permission violation. | 1. Ensure **only one** instance of the dashboard/client is active at a time (duplicate Client IDs cause Mosquitto to disconnect old sockets). <br>2. In Production mode, components must use their exact designated usernames to respect topic security limits. |
| **HTTP API Server Offline** <br>or dashboard shows `[OFFLINE]` | HTTP API server daemon is not running, or ports are blocked by firewall. | 1. Start the HTTP API Server manually in a new window: `python modules/http_api.py`. <br>2. Verify port bounds and settings match `config/config.json` properties. |
| **Images Not Processing** <br>or occupancy stays empty | `images_src/` is empty, or YOLO weight files are missing from the `models/` folder. | 1. Place raw image files in `images_src/` named exactly in `CAM_*.jpg` format (e.g. `CAM_hallway.jpg`). <br>2. Make sure `models/yolo11s.pt` or `models/yolo11m.pt` is downloaded and present. |
| **Diagnostics failing in console** <br>with encoding exceptions | Legacy Windows PowerShell or Command Prompt cannot parse high-contrast Unicode borders. | The configuration wizard includes auto-strip fallbacks. Run the diagnostics inside terminal environments that support `utf-8` formatting, or allow the tool to fallback to clean high-contrast ASCII boundaries. |


---

**Parts of this description were generated by AI. Although efforts have been made for accuracy, please flag inconsistencies if you happen to notice them.**

