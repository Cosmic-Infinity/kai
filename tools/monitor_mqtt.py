"""
MQTT Traffic Monitor

This script subscribes to all KAI topics and displays messages in real-time.
Useful for debugging and monitoring the system.
"""

import paho.mqtt.client as mqtt
import sys
import os
from datetime import datetime

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mqtt_config as config

# Color codes for terminal (Windows PowerShell compatible)
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def colored(text, color):
    """Return colored text if terminal supports it."""
    return f"{color}{text}{Colors.ENDC}"

def on_connect(client, userdata, flags, reason_code, properties):
    """Callback when connected to broker."""
    if reason_code == 0:
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(colored(f"\n[{timestamp}] [OK] Connected to MQTT broker", Colors.GREEN))
        print(colored(f"Broker: {config.MQTT_BROKER_HOST}:{config.MQTT_BROKER_PORT}", Colors.CYAN))
        print(colored("\nSubscribing to topics:", Colors.YELLOW))
        
        # Subscribe to all KAI topics
        topics = [
            (config.TOPIC_FORCE_REQUEST, 1),
            (config.TOPIC_FORCE_SERVED, 1),
            (config.TOPIC_CONTROL, 1),
            (config.TOPIC_POWER, 1),
        ]
        
        for topic, qos in topics:
            client.subscribe(topic, qos)
            print(f"  • {topic}")
        
        print(colored("\n" + "=" * 70, Colors.CYAN))
        print(colored("Monitoring messages (Press Ctrl+C to stop)...", Colors.BOLD))
        print(colored("=" * 70 + "\n", Colors.CYAN))
        
    else:
        print(colored(f"✗ Connection failed (code: {reason_code})", Colors.RED))
        sys.exit(1)

def on_disconnect(client, userdata, flags, reason_code, properties):
    """Callback for disconnection."""
    if reason_code == 0:
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(colored(f"\n[{timestamp}] ✓ Clean disconnect", Colors.GREEN))
    else:
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(colored(f"\n[{timestamp}] ✗ Disconnected unexpectedly (code: {reason_code})", Colors.RED))

def on_message(client, userdata, msg):
    """Callback when a message is received."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    topic = msg.topic
    payload = msg.payload.decode('utf-8', errors='replace')
    
    # Color code based on topic
    if "force_request" in topic:
        topic_color = Colors.YELLOW
        icon = "📤"
    elif "force_served" in topic:
        topic_color = Colors.GREEN
        icon = "✓"
    elif "control" in topic:
        topic_color = Colors.BLUE
        icon = "🎮"
    elif "power" in topic or "POWER" in topic:
        topic_color = Colors.RED
        icon = "⚡"
    else:
        topic_color = Colors.CYAN
        icon = "📨"
    
    # Display message
    print(f"{colored(f'[{timestamp}]', Colors.CYAN)} {icon} {colored(topic, topic_color)}")
    print(f"  → {colored(payload, Colors.BOLD)}")
    print()

def main():
    """Main monitoring loop."""
    print(colored("=" * 70, Colors.HEADER))
    print(colored("KAI IoT System - MQTT Traffic Monitor", Colors.HEADER))
    print(colored("=" * 70, Colors.HEADER))
    
    # Create MQTT client with callback API version 2
    client = mqtt.Client(
        client_id=f"{config.MQTT_CLIENT_ID_PREFIX}_monitor",
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2
    )
    
    # Set callbacks
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_message = on_message
    
    # Set authentication if configured
    if config.MQTT_USERNAME and config.MQTT_PASSWORD:
        client.username_pw_set(config.MQTT_USERNAME, config.MQTT_PASSWORD)
    
    try:
        # Connect to broker
        print(f"\nConnecting to {config.MQTT_BROKER_HOST}:{config.MQTT_BROKER_PORT}...")
        client.connect(config.MQTT_BROKER_HOST, config.MQTT_BROKER_PORT, config.MQTT_KEEPALIVE)
        
        # Start monitoring loop
        client.loop_forever()
        
    except KeyboardInterrupt:
        print(colored("\n\nMonitoring stopped by user.", Colors.YELLOW))
        client.disconnect()
        sys.exit(0)
        
    except ConnectionRefusedError:
        print(colored("\n✗ Connection refused!", Colors.RED))
        print(colored("Make sure Mosquitto is running:", Colors.YELLOW))
        print("  Windows: net start mosquitto")
        print("  Linux:   sudo systemctl start mosquitto")
        sys.exit(1)
        
    except Exception as e:
        print(colored(f"\n✗ Error: {e}", Colors.RED))
        sys.exit(1)

if __name__ == "__main__":
    main()
