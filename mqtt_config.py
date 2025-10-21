"""
MQTT Configuration for the IoT Dashboard/Control System.

This file contains all MQTT broker connection settings and topic definitions.
Modify these settings according to your MQTT broker setup.
"""

# MQTT Broker Settings
MQTT_BROKER_HOST = "localhost"  # Change to your broker's IP/hostname
MQTT_BROKER_PORT = 1883  # Default MQTT port (use 8883 for TLS)
MQTT_KEEPALIVE = 120  # Keepalive interval in seconds (increased for stability)

# Authentication (set to None if broker doesn't require authentication)
MQTT_USERNAME = None  # e.g., "your_username"
MQTT_PASSWORD = None  # e.g., "your_password"

# TLS/SSL Settings (optional, for secure connections)
MQTT_USE_TLS = False
MQTT_CA_CERTS = None  # Path to CA certificate file
MQTT_CERTFILE = None  # Path to client certificate
MQTT_KEYFILE = None   # Path to client key

# Quality of Service (QoS) levels
# QoS 0: At most once delivery (fire and forget)
# QoS 1: At least once delivery (acknowledged delivery)
# QoS 2: Exactly once delivery (assured delivery)
MQTT_QOS = 1  # Recommended: QoS 1 for reliable delivery

# Topic Definitions
# These topics correspond to the feeds in the original system
TOPIC_FORCE_REQUEST = "kai/force_request"
TOPIC_FORCE_SERVED = "kai/force_served"
TOPIC_CONTROL = "kai/control"
TOPIC_POWER = "kai/power"

# Message Retention
# If True, broker will retain the last message sent to each topic
MQTT_RETAIN = False

# Connection Retry Settings
MQTT_RECONNECT_DELAY = 2  # Seconds to wait before reconnecting (reduced for faster recovery)
MQTT_MAX_RECONNECT_ATTEMPTS = 0  # Maximum reconnection attempts (0 = infinite)

# Client ID Prefix (will be appended with component name)
MQTT_CLIENT_ID_PREFIX = "kai"

# Logging
MQTT_DEBUG = False  # Set to True for verbose MQTT logging
