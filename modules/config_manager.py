import json
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.json")

DEFAULT_CONFIG = {
    "IMAGE_SERVER_INTERVAL": 60,
    "DASHBOARD_INTERVAL": 30,
    "CONTROL_SERVER_INTERVAL": 30,
    "INACTIVITY_THRESHOLD": 10,
    "MQTT_KEEPALIVE": 120,
    "MQTT_RECONNECT_DELAY": 2
}

def load_config():
    if not os.path.exists(CONFIG_PATH):
        save_config(DEFAULT_CONFIG)
        return DEFAULT_CONFIG.copy()
    try:
        with open(CONFIG_PATH, 'r') as f:
            config = json.load(f)
            # Merge with defaults
            for k, v in DEFAULT_CONFIG.items():
                if k not in config:
                    config[k] = v
            return config
    except:
        return DEFAULT_CONFIG.copy()

def save_config(config_dict):
    try:
        with open(CONFIG_PATH, 'w') as f:
            json.dump(config_dict, f, indent=4)
    except Exception as e:
        print(f"Failed to save config: {e}")
