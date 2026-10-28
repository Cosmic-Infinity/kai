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
from config_manager import load_config, load_power_state, save_power_state, load_frame_log, frame_stale_after
from logger import kai_print as print


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
        print(f"[Control] [FAST] Instantly processed command: {command}")


_power_state: Dict[str, str] = load_power_state()
_power_state_lock = threading.Lock()


def _record_power_state(content: str) -> None:
    """Remember the last published state per camera so the HTTP API can report it to dashboards."""
    camera_id, state = content.rsplit("_", 1)
    with _power_state_lock:
        _power_state[camera_id] = state
        try:
            save_power_state(_power_state)
        except OSError as e:
            print(f"[Power] Could not persist power state: {e}")


def write_to_power_feed(content: str) -> None:
    """Writes a command to the POWER feed."""
    if not append_message(POWER_FEED, content):
        print(f"[Power] Failed to write '{content}' to {POWER_FEED}")
        return
    _record_power_state(content)
    print(f"[Power] Wrote '{content}' to {POWER_FEED}")


def control_command_loop() -> None:
    """Checks for UI control commands and responds instantly."""
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
    Monitors camera statuses and turns off power after consecutive 'NO' statuses.

    This loop runs on its own timer (CONTROL_SERVER_INTERVAL), independent of the image
    server's capture timer: every check reads whatever status is current, so the same frame
    can be counted by several checks and the auto-off delay is
    INACTIVITY_THRESHOLD x CONTROL_SERVER_INTERVAL. Frames the image server hasn't refreshed
    recently are not evidence of anything - counting pauses for them until fresh frames arrive.
    """
    print(f"[Monitoring Loop] Started - checking camera statuses based on dynamic configuration...")
    camera_inactivity_count: Dict[str, int] = {}
    stale_cameras = set()
    last_check_time = 0

    while True:
        try:
            config = load_config()
            interval = config.get("CONTROL_SERVER_INTERVAL", 30)
            threshold = config.get("INACTIVITY_THRESHOLD", 10)
            
            current_time = time.time()
            if current_time - last_check_time >= interval:
                statuses = get_camera_status()
                frames = load_frame_log()
                stale_after = frame_stale_after(config)
                for cam_id, status in statuses.items():
                    processed_at = frames.get(cam_id, {}).get("processed_at")
                    if processed_at is None or current_time - processed_at > stale_after:
                        if cam_id not in stale_cameras:
                            age = "never processed" if processed_at is None else f"last processed {int(current_time - processed_at)}s ago"
                            print(f"[Monitoring Loop] {cam_id}: frame is stale ({age}, limit {stale_after}s) - "
                                  f"holding inactivity count at {camera_inactivity_count.get(cam_id, 0)}. Is the Image Server running?")
                            stale_cameras.add(cam_id)
                        continue
                    if cam_id in stale_cameras:
                        print(f"[Monitoring Loop] {cam_id}: fresh frames again - resuming inactivity tracking.")
                        stale_cameras.discard(cam_id)

                    if status == "NO":
                        camera_inactivity_count[cam_id] = camera_inactivity_count.get(cam_id, 0) + 1
                    else:
                        camera_inactivity_count[cam_id] = 0

                    if camera_inactivity_count.get(cam_id, 0) >= threshold:
                        write_to_power_feed(f"{cam_id}_OFF")
                        print(f"[Inactivity] Turned off {cam_id} due to {threshold} consecutive 'NO' statuses.")
                        camera_inactivity_count[cam_id] = 0
                
                last_check_time = current_time
            
            time.sleep(1)
        except Exception as e:
            print(f"[Monitoring Loop] Error: {e}")
            time.sleep(1)


def main() -> None:
    """Main entry point - starts both control and monitoring threads."""
    # Explicitly configure the module role for secure isolated authentication
    from feeds import configure_role
    configure_role("control_server")
    
    print("=" * 60)
    print("Control Server started.")
    print("=" * 60)
    print(f"[FAST] Control commands: Instant response (checks every 0.1s)")
    print(f"[CONF] Using dynamic config from config.json")
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
