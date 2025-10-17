import os
import time
from typing import Dict

from feeds import append_message, consume_messages

IMAGE_DIR = "images_ready"
CONTROL_FEED = "control"
POWER_FEED = "POWER"
REFRESH_INTERVAL = 30  # seconds
INACTIVITY_THRESHOLD = 10  # 10 consecutive "NO" statuses


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
    """Reads and processes commands from the control feed stub."""
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
        print(f"[Control] Processed command: {command}")


def write_to_power_feed(content: str) -> None:
    """Writes a command to the POWER feed stub."""
    append_message(POWER_FEED, content)
    print(f"[Power] Wrote '{content}' to {POWER_FEED}")


def main() -> None:
    """Main loop for the control server."""
    print("Control Server started.")
    camera_inactivity_count: Dict[str, int] = {}

    while True:
        read_control_feed()

        statuses = get_camera_status()
        for cam_id, status in statuses.items():
            if status == "NO":
                camera_inactivity_count[cam_id] = camera_inactivity_count.get(cam_id, 0) + 1
            else:
                camera_inactivity_count[cam_id] = 0

            if camera_inactivity_count.get(cam_id, 0) >= INACTIVITY_THRESHOLD:
                write_to_power_feed(f"{cam_id}_OFF")
                print(f"[Inactivity] Turned off {cam_id} due to inactivity.")
                camera_inactivity_count[cam_id] = 0

        time.sleep(REFRESH_INTERVAL)


if __name__ == "__main__":
    main()
