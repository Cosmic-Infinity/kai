"""MQTT-based feed system for the IoT Dashboard/Control System.

This module provides MQTT publish/subscribe functionality with the same API
as the original file-based system, allowing seamless transition from file stubs
to a real MQTT broker.

Each feed is now represented as an MQTT topic. The system uses a shared MQTT
client with automatic reconnection and message queuing.
"""

from __future__ import annotations

import time
import threading
import uuid
from collections import defaultdict, deque
from typing import Iterable, List, Dict, Callable, Optional

import paho.mqtt.client as mqtt

from config_manager import get_config_namespace
config = get_config_namespace()


class MQTTFeedManager:
    """
    Manages MQTT connections and message handling for the feed system.
    
    This class implements a singleton pattern to ensure all components
    share the same MQTT connection and message queues.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._initialized = True
        self._client: Optional[mqtt.Client] = None
        self._connected = False
        self._message_queues: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._subscriptions: Dict[str, List[Callable]] = defaultdict(list)
        self._lock = threading.RLock()
        self._connect_thread = None
        
        self._topic_map = {
            "force_request": config.TOPIC_FORCE_REQUEST,
            "force_served": config.TOPIC_FORCE_SERVED,
            "control": config.TOPIC_CONTROL,
            "POWER": config.TOPIC_POWER,
        }
        
        self.host = config.MQTT_BROKER_HOST
        self.port = config.MQTT_BROKER_PORT
        
        self._initialize_client()
        
    def configure(self, host: str, port: int, username: str = None, password: str = None):
        self.host = host
        self.port = port
        config.MQTT_BROKER_HOST = host
        config.MQTT_BROKER_PORT = port
        if username:
            config.MQTT_USERNAME = username
            config.MQTT_PASSWORD = password
            if self._client:
                self._client.username_pw_set(username, password)
    
    def _initialize_client(self):
        """Initialize the MQTT client with configuration."""
        # Use timestamp + UUID for truly unique client IDs to prevent collisions
        import time as _time
        client_id = f"{config.MQTT_CLIENT_ID_PREFIX}_{int(_time.time() * 1000)}_{uuid.uuid4().hex[:6]}"
        
        # Create client with clean session and handle paho-mqtt 1.x and 2.x compatibility
        try:
            self._client = mqtt.Client(
                client_id=client_id, 
                clean_session=True,
                callback_api_version=mqtt.CallbackAPIVersion.VERSION2
            )
        except (AttributeError, TypeError):
            self._client = mqtt.Client(
                client_id=client_id, 
                clean_session=True
            )
        
        # Set callbacks
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message = self._on_message
        
        # Set authentication if configured
        if config.MQTT_USERNAME and config.MQTT_PASSWORD:
            self._client.username_pw_set(config.MQTT_USERNAME, config.MQTT_PASSWORD)
        
        # Set TLS if configured
        if config.MQTT_USE_TLS:
            self._client.tls_set(
                ca_certs=config.MQTT_CA_CERTS,
                certfile=config.MQTT_CERTFILE,
                keyfile=config.MQTT_KEYFILE
            )
        
        # Enable logging if debug mode
        if config.MQTT_DEBUG:
            self._client.enable_logger()
        
        # Set reconnect delay
        self._client.reconnect_delay_set(min_delay=1, max_delay=30)
    
    def _ensure_connection_started(self):
        """Ensure that the background connection thread has been started."""
        if not self._connect_thread:
            self._start_connection()

    def start(self):
        """Force a reconnect using the current configuration."""
        self._connected = False
        if self._client:
            try:
                self._client.disconnect()
            except Exception as e:
                print(f"[MQTT] Disconnect error: {e}")
        self._start_connection()

    def _start_connection(self):
        """Start MQTT connection in a background thread."""
        if self._connect_thread and self._connect_thread.is_alive():
            return
        
        self._connect_thread = threading.Thread(target=self._connect_loop, daemon=True)
        self._connect_thread.start()
    
    def _connect_loop(self):
        """Connection loop with retry logic."""
        attempt = 0
        from config_manager import load_config
        
        while True:
            try:
                dyn_config = load_config()
                keepalive = dyn_config.get("MQTT_KEEPALIVE", config.MQTT_KEEPALIVE)
                reconnect_delay = dyn_config.get("MQTT_RECONNECT_DELAY", config.MQTT_RECONNECT_DELAY)
                
                if not self._connected:
                    if config.MQTT_MAX_RECONNECT_ATTEMPTS > 0 and attempt >= config.MQTT_MAX_RECONNECT_ATTEMPTS:
                        print(f"[MQTT] Max reconnection attempts ({config.MQTT_MAX_RECONNECT_ATTEMPTS}) reached.")
                        break
                    
                    attempt += 1
                    if attempt == 1:
                        auth_info = f"user={config.MQTT_USERNAME}" if config.MQTT_USERNAME else "no auth"
                        print(f"[MQTT] Connecting to broker at {config.MQTT_BROKER_HOST}:{config.MQTT_BROKER_PORT} ({auth_info})...")
                    else:
                        print(f"[MQTT] Reconnection attempt {attempt}...")
                    
                    try:
                        self._client.loop_stop()
                        self._client.connect(
                            config.MQTT_BROKER_HOST,
                            config.MQTT_BROKER_PORT,
                            keepalive
                        )
                        
                        self._client.loop_start()
                    except ConnectionRefusedError:
                        print(f"[MQTT] Connection refused. Is the broker running? Retrying in {reconnect_delay}s...")
                        time.sleep(reconnect_delay)
                        continue
                    except Exception as conn_err:
                        print(f"[MQTT] Connection error: {conn_err}. Retrying in {reconnect_delay}s...")
                        time.sleep(reconnect_delay)
                        continue
                    
                    # Wait for connection with timeout
                    for _ in range(30):  # Wait up to 3 seconds
                        if self._connected:
                            break
                        time.sleep(0.1)
                    
                    if not self._connected:
                        print(f"[MQTT] Connection timeout. Retrying in {reconnect_delay}s...")
                        time.sleep(reconnect_delay)
                else:
                    # Connected, reset attempt counter
                    attempt = 0
                    time.sleep(1)
            except Exception as e:
                print(f"[MQTT] Unexpected error in connection loop: {e}")
                time.sleep(reconnect_delay)
    
    def _on_connect(self, client, userdata, flags, *args, **kwargs):
        """Callback for when the client connects to the broker."""
        reason_code = args[0] if args else kwargs.get('reason_code', kwargs.get('rc', 0))
        if reason_code == 0:
            print("[MQTT] Successfully connected to broker.")
            self._connected = True
            
            # Subscribe to all topics we need to listen to
            topics_to_subscribe = set(self._topic_map.values())
            for topic in topics_to_subscribe:
                client.subscribe(topic, qos=config.MQTT_QOS)
                print(f"[MQTT] Subscribed to topic: {topic}")
        else:
            print(f"[MQTT] Connection failed with code {reason_code}")
            self._connected = False
            
    def _on_disconnect(self, client, userdata, *args, **kwargs):
        """Callback for when the client disconnects from the broker."""
        self._connected = False
        # Extract reason code (rc) compatibly for paho-mqtt 1.x and 2.x
        reason_code = 0
        if args:
            if len(args) >= 3:
                reason_code = args[1]  # paho-mqtt 2.x: (flags, reason_code, properties)
            else:
                reason_code = args[0]  # paho-mqtt 1.x: (rc)
        else:
            reason_code = kwargs.get('reason_code', kwargs.get('rc', 0))

        if reason_code != 0:
            # Error codes: 1=protocol, 2=client ID, 3=server unavailable, 4=bad auth, 5=not authorized, 7=network
            error_msgs = {
                1: "Protocol version error",
                2: "Client identifier rejected",
                3: "Server unavailable",
                4: "Bad username or password",
                5: "Not authorized",
                7: "Network error - connection lost"
            }
            error_detail = error_msgs.get(reason_code, f"Unknown error (code {reason_code})")
            print(f"[MQTT] Disconnected: {error_detail}. Reconnecting...")
        else:
            print(f"[MQTT] Clean disconnect.")
    
    def _on_message(self, client, userdata, msg):
        """Callback for when a message is received."""
        try:
            payload = msg.payload.decode('utf-8').strip()
            topic = msg.topic
            
            if config.MQTT_DEBUG:
                print(f"[MQTT] Received message on {topic}: {payload}")
            
            # Add to appropriate queue
            with self._lock:
                self._message_queues[topic].append(payload)
        except Exception as e:
            print(f"[MQTT] Error processing message: {e}")
    
    def _get_topic(self, feed_name: str) -> str:
        """Get MQTT topic for a feed name."""
        return self._topic_map.get(feed_name, f"kai/{feed_name}")
    
    def publish(self, feed_name: str, message: str) -> bool:
        """
        Publish a message to a feed (topic).
        
        Args:
            feed_name: Name of the feed (will be mapped to MQTT topic)
            message: Message to publish
            
        Returns:
            True if message was published successfully, False otherwise
        """
        self._ensure_connection_started()
        if not self._connected:
            print(f"[MQTT] Not connected (broker={config.MQTT_BROKER_HOST}:{config.MQTT_BROKER_PORT}). Cannot publish to {feed_name}: {message}")
            return False
        
        topic = self._get_topic(feed_name)
        try:
            result = self._client.publish(
                topic,
                message,
                qos=config.MQTT_QOS,
                retain=config.MQTT_RETAIN
            )
            
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                if config.MQTT_DEBUG:
                    print(f"[MQTT] Published to {topic}: {message}")
                return True
            else:
                print(f"[MQTT] Failed to publish to {topic}: {result.rc}")
                return False
        except Exception as e:
            print(f"[MQTT] Error publishing to {feed_name}: {e}")
            return False
    
    def consume(self, feed_name: str) -> List[str]:
        """
        Consume all pending messages from a feed and clear the queue.
        
        Args:
            feed_name: Name of the feed
            
        Returns:
            List of messages (strings)
        """
        self._ensure_connection_started()
        topic = self._get_topic(feed_name)
        with self._lock:
            messages = list(self._message_queues[topic])
            self._message_queues[topic].clear()
        return messages
    
    def peek(self, feed_name: str) -> List[str]:
        """
        Peek at pending messages without consuming them.
        
        Args:
            feed_name: Name of the feed
            
        Returns:
            List of messages (strings)
        """
        self._ensure_connection_started()
        topic = self._get_topic(feed_name)
        with self._lock:
            return list(self._message_queues[topic])
    
    def clear(self, feed_name: str):
        """
        Clear all pending messages from a feed.
        
        Args:
            feed_name: Name of the feed
        """
        self._ensure_connection_started()
        topic = self._get_topic(feed_name)
        with self._lock:
            self._message_queues[topic].clear()
    
    def is_connected(self) -> bool:
        """Check if connected to MQTT broker."""
        self._ensure_connection_started()
        return self._connected
    
    def disconnect(self):
        """Disconnect from the MQTT broker."""
        if self._client:
            self._client.loop_stop()
            self._client.disconnect()
            self._connected = False


# Global manager instance
_manager = MQTTFeedManager()
feed_manager = _manager

# Public API - maintains compatibility with file-based system
def append_message(feed_name: str, message: str) -> None:
    """Publish a single message to the specified feed."""
    _manager.publish(feed_name, message)


def append_messages(feed_name: str, messages: Iterable[str]) -> None:
    """Publish multiple messages to the specified feed."""
    for message in messages:
        _manager.publish(feed_name, str(message))


def consume_messages(feed_name: str) -> List[str]:
    """Return all pending messages for the feed and clear it."""
    return _manager.consume(feed_name)


def peek_messages(feed_name: str) -> List[str]:
    """Return all pending messages for the feed without clearing it."""
    return _manager.peek(feed_name)


def clear_feed(feed_name: str) -> None:
    """Remove all pending messages from the feed."""
    _manager.clear(feed_name)


def is_connected() -> bool:
    """Check if connected to MQTT broker."""
    return _manager.is_connected()


def disconnect() -> None:
    """Disconnect from MQTT broker."""
    _manager.disconnect()
