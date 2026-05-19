import os
import sys
import time
import threading
from typing import Dict

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feeds import append_message, consume_messages

# Use absolute paths based on project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_DIR = os.path.join(PROJECT_ROOT, "images_ready")
CONTROL_FEED = "control"
POWER_FEED = "POWER"
from config_manager import load_config


def _parse_status_filename(filename: str):
    stem, ext = os.path.splitext(filename)
    if ext.lower() not in {".jpg", ".jpeg", ".png"}:
        return None
    if "_" not in stem:
        return None
    camera_id, status = stem.rsplit("_", 1)
    status = status.upper()
    if not camera_id.startswith("CAM_"):
        return None
    if status not in {"YES", "NO"}:
        return None
    return camera_id, status


def get_camera_status() -> Dict[str, str]:
    """Reads camera statuses from the ready image directory."""
    statuses: Dict[str, str] = {}
    try:
        for filename in os.listdir(IMAGE_DIR):
            parsed = _parse_status_filename(filename)
            if parsed:
                camera_id, status = parsed
                statuses[camera_id] = status
    except FileNotFoundError:
        print(f"Image directory '{IMAGE_DIR}' not found.")
    return statuses


def read_control_feed() -> None:
    """Reads and processes commands from the control feed - responds instantly."""
    commands = consume_messages(CONTROL_FEED)
    for command in commands:
        command = command.strip()
        if not command.startswith("SET_CAM_"):
            print(f"[Control] Ignoring unknown command '{command}'.")
            continue
        try:
            target, desired_state = command[len("SET_") :].rsplit("_", 1)
        except ValueError:
            print(f"[Control] Malformed command '{command}'.")
            continue
        desired_state = desired_state.upper()
        if desired_state not in {"ON", "OFF"}:
            print(f"[Control] Invalid state '{desired_state}' in '{command}'.")
            continue
        write_to_power_feed(f"{target}_{desired_state}")
        print(f"[Control] ⚡ Instantly processed command: {command}")


def write_to_power_feed(content: str) -> None:
    """Writes a command to the POWER feed."""
    append_message(POWER_FEED, content)
    print(f"[Power] Wrote '{content}' to {POWER_FEED}")


def control_command_loop() -> None:
    """
    Fast loop that continuously checks for control commands from the UI.
    Runs in a separate thread to respond instantly to user requests.
    """
    print("[Control Loop] Started - monitoring control feed for instant response...")
    while True:
        try:
            read_control_feed()
            time.sleep(0.1)  # Check every 100ms for instant response
        except Exception as e:
            print(f"[Control Loop] Error: {e}")
            time.sleep(1)


def camera_monitoring_loop() -> None:
    """
    Slow loop that monitors camera statuses every 30 seconds.
    Automatically turns off power after 10 consecutive "NO" statuses.
    """
    print("[Monitoring Loop] Started - checking camera statuses every 30s...")
    camera_inactivity_count: Dict[str, int] = {}

    while True:
        try:
            config = load_config()
            interval = config.get("CONTROL_SERVER_INTERVAL", 30)
            threshold = config.get("INACTIVITY_THRESHOLD", 10)
            
            statuses = get_camera_status()
            for cam_id, status in statuses.items():
                if status == "NO":
                    camera_inactivity_count[cam_id] = camera_inactivity_count.get(cam_id, 0) + 1
                else:
                    camera_inactivity_count[cam_id] = 0

                if camera_inactivity_count.get(cam_id, 0) >= threshold:
                    write_to_power_feed(f"{cam_id}_OFF")
                    print(f"[Inactivity] Turned off {cam_id} due to {threshold} consecutive 'NO' statuses.")
                    camera_inactivity_count[cam_id] = 0
            
            time.sleep(interval)
        except Exception as e:
            print(f"[Monitoring Loop] Error: {e}")
            time.sleep(5)


def main() -> None:
    """Main entry point - starts both control and monitoring threads."""
    print("=" * 60)
    print("Control Server started.")
    print("=" * 60)
    print(f"⚡ Control commands: Instant response (checks every 0.1s)")
    print(f"📊 Using dynamic config from config.json")
    print("=" * 60)
    
    # Create threads for parallel processing
    control_thread = threading.Thread(target=control_command_loop, daemon=True, name="ControlThread")
    monitoring_thread = threading.Thread(target=camera_monitoring_loop, daemon=True, name="MonitoringThread")
    
    # Start both threads
    control_thread.start()
    monitoring_thread.start()
    
    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\nControl Server stopped by user.")


if __name__ == "__main__":
    main()
