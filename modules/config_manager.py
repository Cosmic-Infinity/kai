import json
import os
import sys
import time

MODULE_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(MODULE_ROOT)
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "config.json")
# Last known power state per camera, written by the control server and served by the HTTP API
POWER_STATE_PATH = os.path.join(PROJECT_ROOT, "state", "power_state.json")
# Per-camera frame timestamps, written by the image server:
# {camera_id: {"captured_at": epoch, "camera_clock_offset": seconds|None, "processed_at": epoch (server clock)}}
FRAME_LOG_PATH = os.path.join(PROJECT_ROOT, "state", "frames.json")

DEFAULT_CONFIG = {
    "IMAGE_SERVER_INTERVAL": 60,
    "DASHBOARD_INTERVAL": 30,
    "CONTROL_SERVER_INTERVAL": 30,
    "INACTIVITY_THRESHOLD": 10,
    "MQTT_KEEPALIVE": 120,
    "MQTT_RECONNECT_DELAY": 2,
    "MQTT_BROKER_HOST": "127.0.0.1",
    "MQTT_BROKER_PORT": 1883,
    "MQTT_USERNAME": "kai_admin",
    "MQTT_PASSWORD": "kai",
    "API_KEY": "testtest",
    "TOPIC_FORCE_REQUEST": "kai/force_request",
    "TOPIC_FORCE_SERVED": "kai/force_served",
    "TOPIC_CONTROL": "kai/control",
    "TOPIC_POWER": "kai/power",
    "MQTT_QOS": 1,
    "MQTT_RETAIN": False,
    "MQTT_MAX_RECONNECT_ATTEMPTS": 0,
    "MQTT_CLIENT_ID_PREFIX": "kai",
    "MQTT_DEBUG": False
}

# Last successfully parsed config, reused if the file is briefly unreadable (e.g. mid-edit)
_last_good_config = None
_last_reported_error = None


def atomic_write_json(path, data):
    """Write JSON to `path` so concurrent readers never see a half-written file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.{os.getpid()}.tmp"
    try:
        with open(tmp_path, 'w') as f:
            json.dump(data, f, indent=4)
        # On Windows os.replace fails while another process has the target open; retry briefly
        for attempt in range(10):
            try:
                os.replace(tmp_path, path)
                return
            except PermissionError:
                if attempt == 9:
                    raise
                time.sleep(0.05)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def load_config():
    """
    Load config.json merged over the defaults.

    A missing file is created from the defaults. An unreadable/corrupt file is reported
    on stderr; the last good config is reused if there is one, otherwise the error is
    raised so a service never silently starts with the built-in default credentials.
    """
    global _last_good_config, _last_reported_error

    if not os.path.exists(CONFIG_PATH):
        try:
            save_config(DEFAULT_CONFIG)
        except OSError as e:
            print(f"[Config] Could not create {CONFIG_PATH}: {e}", file=sys.stderr)
        return DEFAULT_CONFIG.copy()

    try:
        with open(CONFIG_PATH, 'r') as f:
            config = json.load(f)
    except (OSError, ValueError) as e:
        message = f"[Config] Failed to read {CONFIG_PATH}: {e}"
        if _last_good_config is not None:
            message += " - continuing with the last successfully loaded configuration."
        if message != _last_reported_error:  # callers poll every second; report each distinct failure once
            print(message, file=sys.stderr)
            _last_reported_error = message
        if _last_good_config is None:
            raise
        return dict(_last_good_config)

    # Merge with defaults
    for k, v in DEFAULT_CONFIG.items():
        if k not in config:
            config[k] = v
    _last_good_config = dict(config)
    _last_reported_error = None
    return config


def save_config(config_dict):
    """Persist the config. Raises OSError if it cannot be written."""
    atomic_write_json(CONFIG_PATH, config_dict)


def _load_state_file(path):
    """Read a JSON state file, returning {} if it doesn't exist yet or can't be read (the error is reported)."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except (OSError, ValueError) as e:
        print(f"[Config] Failed to read {path}: {e}", file=sys.stderr)
        return {}


def load_power_state():
    """Return {camera_id: "ON"|"OFF"} as last recorded by the control server (empty if none yet)."""
    return _load_state_file(POWER_STATE_PATH)


def save_power_state(state):
    """Persist the power state map. Raises OSError if it cannot be written."""
    atomic_write_json(POWER_STATE_PATH, state)


def load_frame_log():
    """Return {camera_id: {"captured_at": epoch, "processed_at": epoch}} as last recorded by the image server."""
    return _load_state_file(FRAME_LOG_PATH)


def save_frame_log(frames):
    """Persist the frame timestamp log. Raises OSError if it cannot be written."""
    atomic_write_json(FRAME_LOG_PATH, frames)


def frame_stale_after(config):
    """
    Seconds after which a camera's last processed frame no longer counts as current evidence.
    Three missed capture cycles: tolerant of slow batches, but catches a stopped image server.
    """
    return 3 * config.get("IMAGE_SERVER_INTERVAL", 60)


class ConfigNamespace:
    def __init__(self, dictionary):
        self.__dict__.update(dictionary)
    def __getattr__(self, name):
        return self.__dict__.get(name, None)

def get_config_namespace():
    return ConfigNamespace(load_config())
