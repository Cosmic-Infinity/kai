import json
import os

MODULE_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(MODULE_ROOT)
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "config.json")

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

def load_config():
    if not os.path.exists(CONFIG_PATH):
        os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
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
        os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
        with open(CONFIG_PATH, 'w') as f:
            json.dump(config_dict, f, indent=4)
    except Exception as e:
        print(f"Failed to save config: {e}")

class ConfigNamespace:
    def __init__(self, dictionary):
        self.__dict__.update(dictionary)
    def __getattr__(self, name):
        return self.__dict__.get(name, None)

def get_config_namespace():
    return ConfigNamespace(load_config())
