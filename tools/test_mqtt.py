"""
Simple MQTT connection test script.

Run this script to verify that your MQTT broker is running and accessible.
"""

import paho.mqtt.client as mqtt
import time
import sys

BROKER_HOST = "localhost"
BROKER_PORT = 1883

def on_connect(client, userdata, flags, reason_code, properties):
    """Callback when connection attempt completes."""
    if reason_code == 0:
        print("[OK] Successfully connected to MQTT broker!")
        print(f"  Broker: {BROKER_HOST}:{BROKER_PORT}")
        print("\nTesting publish/subscribe...")
        
        # Subscribe to test topic
        client.subscribe("kai/test", qos=1)
        
        # Publish test message
        client.publish("kai/test", "Hello MQTT!", qos=1)
        
    else:
        error_messages = {
            1: "Connection refused - incorrect protocol version",
            2: "Connection refused - invalid client identifier",
            3: "Connection refused - server unavailable",
            4: "Connection refused - bad username or password",
            5: "Connection refused - not authorized"
        }
        print(f"✗ Connection failed!")
        print(f"  Error code: {reason_code}")
        print(f"  Error: {error_messages.get(reason_code, 'Unknown error')}")
        sys.exit(1)

def on_disconnect(client, userdata, flags, reason_code, properties):
    """Callback when disconnected."""
    if reason_code != 0:
        print(f"✗ Unexpected disconnection (code: {reason_code})")

def on_message(client, userdata, msg):
    """Callback when a message is received."""
    payload = msg.payload.decode('utf-8')
    print(f"✓ Message received successfully!")
    print(f"  Topic: {msg.topic}")
    print(f"  Payload: {payload}")
    print("\n✓ All tests passed! MQTT broker is working correctly.")
    
    # Disconnect after successful test
    client.disconnect()

def main():
    print("=" * 60)
    print("MQTT Broker Connection Test")
    print("=" * 60)
    print(f"\nAttempting to connect to broker at {BROKER_HOST}:{BROKER_PORT}...")
    
    # Create client with callback API version 2
    client = mqtt.Client(client_id="kai_test_client", callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_message = on_message
    
    try:
        # Connect to broker
        client.connect(BROKER_HOST, BROKER_PORT, 60)
        
        # Start loop with timeout
        client.loop_start()
        
        # Wait for messages
        time.sleep(3)
        
        # Stop loop
        client.loop_stop()
        
    except ConnectionRefusedError:
        print("\n✗ Connection refused!")
        print("  Make sure the MQTT broker (Mosquitto) is running.")
        print("\n  Windows: net start mosquitto")
        print("  Linux:   sudo systemctl start mosquitto")
        print("  Mac:     brew services start mosquitto")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n✗ Error connecting to broker: {e}")
        print(f"  Type: {type(e).__name__}")
        sys.exit(1)

if __name__ == "__main__":
    main()
