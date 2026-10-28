import os
import sys
import json
import hmac
import logging
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse

# Setup path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_manager import load_config, save_config, load_power_state, load_frame_log

from logger import configure_logging
configure_logging(logging.INFO)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_DIR = os.path.join(PROJECT_ROOT, "images_ready")

def discover_cameras():
    cameras = {}
    power_states = load_power_state()
    frames = load_frame_log()
    try:
        for filename in os.listdir(IMAGE_DIR):
            stem, ext = os.path.splitext(filename)
            if ext.lower() not in {".jpg", ".jpeg", ".png"}: continue
            if "_" not in stem: continue
            camera_id, status = stem.rsplit("_", 1)
            status = status.upper()
            if not camera_id.startswith("CAM_"): continue
            if status not in {"YES", "NO"}: continue
            cameras[camera_id] = {
                "image_path": f"/images/{filename}",
                "status": status,
                "bboxes": []
            }
            # Last power state published by the control server (omitted until one exists)
            if camera_id in power_states:
                cameras[camera_id]["power_state"] = power_states[camera_id]
            # Epoch seconds: when the frame was taken, and when the image server last analysed it
            if camera_id in frames:
                cameras[camera_id]["captured_at"] = frames[camera_id].get("captured_at")
                cameras[camera_id]["processed_at"] = frames[camera_id].get("processed_at")
                cameras[camera_id]["camera_clock_offset"] = frames[camera_id].get("camera_clock_offset")
            # Attempt to load bboxes
            bbox_path = os.path.join(IMAGE_DIR, f"{camera_id}_bboxes.json")
            if os.path.exists(bbox_path):
                try:
                    with open(bbox_path, 'r') as f:
                        cameras[camera_id]["bboxes"] = json.load(f)
                except (OSError, ValueError) as e:
                    logging.warning(f"Could not read bounding boxes for {camera_id}: {e}")
    except FileNotFoundError:
        logging.warning(f"Image directory '{IMAGE_DIR}' not found; no cameras to report.")
    return cameras

class KAIRequestHandler(SimpleHTTPRequestHandler):
    def is_authorized(self):
        config = load_config()
        server_key = config.get("API_KEY", "")
        if not server_key:
            return True
        
        client_key = self.headers.get("X-API-Key", "")
        if not client_key:
            auth_header = self.headers.get("Authorization", "")
            if auth_header.startswith("Bearer "):
                client_key = auth_header[7:].strip()
                
        return hmac.compare_digest(client_key.encode('utf-8'), server_key.encode('utf-8'))

    def send_unauthorized(self):
        self.send_response(401)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps({"error": "Unauthorized - Invalid X-API-Key"}).encode('utf-8'))

    def do_GET(self):
        if not self.is_authorized():
            self.send_unauthorized()
            return
            
        parsed = urlparse(self.path)
        if parsed.path == "/api/cameras":
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            cameras = discover_cameras()
            self.wfile.write(json.dumps(cameras).encode('utf-8'))
        elif parsed.path == "/api/config":
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            # Whitelist of client-safe configuration preferences (timers & layout)
            ALLOWED_FIELDS = {
                "IMAGE_SERVER_INTERVAL",
                "DASHBOARD_INTERVAL",
                "CONTROL_SERVER_INTERVAL",
                "INACTIVITY_THRESHOLD",
                "MQTT_KEEPALIVE",
                "MQTT_RECONNECT_DELAY",
                "MQTT_QOS",
                "MQTT_RETAIN",
                "MQTT_DEBUG"
            }
            
            current_config = load_config()
            # Only send client-safe keys over the network
            filtered_config = {k: v for k, v in current_config.items() if k in ALLOWED_FIELDS}
            self.wfile.write(json.dumps(filtered_config).encode('utf-8'))

        elif parsed.path.startswith("/images/"):
            super().do_GET()
        else:
            self.send_error(404, "Not Found")

    def translate_path(self, path):
        parsed = urlparse(path)
        if parsed.path.startswith("/images/"):
            filename = os.path.basename(parsed.path)
            return os.path.join(IMAGE_DIR, filename)
        return super().translate_path(path)

    def do_POST(self):
        if not self.is_authorized():
            self.send_unauthorized()
            return
            
        parsed = urlparse(self.path)
        if parsed.path == "/api/config":
            try:
                content_length = int(self.headers.get('Content-Length') or 0)
                post_data = self.rfile.read(content_length)
                updates = json.loads(post_data.decode('utf-8'))
                logging.info(f"UI submitted configuration updates via HTTP POST: {updates}")
                
                # Strict whitelist of modifiable keys, along with expected type and numeric range limits
                ALLOWED_FIELDS = {
                    "IMAGE_SERVER_INTERVAL": (int, 1, 3600),
                    "DASHBOARD_INTERVAL": (int, 1, 3600),
                    "CONTROL_SERVER_INTERVAL": (int, 1, 3600),
                    "INACTIVITY_THRESHOLD": (int, 1, 3600),
                    "MQTT_KEEPALIVE": (int, 5, 3600),
                    "MQTT_RECONNECT_DELAY": (int, 1, 60),
                    "MQTT_QOS": (int, 0, 2),
                    "MQTT_RETAIN": (bool, None, None),
                    "MQTT_DEBUG": (bool, None, None)
                }
                
                current_config = load_config()
                
                # 1. Enforce strict parameter validation
                for key, val in updates.items():
                    if key not in ALLOWED_FIELDS:
                        logging.warning(f"Rejection: Client attempted to modify protected or invalid key: {key}")
                        self.send_response(400)
                        self.send_header('Content-Type', 'application/json')
                        self.send_header('Access-Control-Allow-Origin', '*')
                        self.end_headers()
                        self.wfile.write(json.dumps({"error": f"Access Denied: Key '{key}' is protected or invalid."}).encode('utf-8'))
                        return
                        
                    expected_type, min_val, max_val = ALLOWED_FIELDS[key]
                    
                    # Verify type of the parameter
                    if not isinstance(val, expected_type) or isinstance(val, bool) != (expected_type is bool):
                        logging.warning(f"Rejection: Client sent invalid type for key {key}. Expected {expected_type.__name__}, got {type(val).__name__}")
                        self.send_response(400)
                        self.send_header('Content-Type', 'application/json')
                        self.send_header('Access-Control-Allow-Origin', '*')
                        self.end_headers()
                        self.wfile.write(json.dumps({"error": f"Type Error: Key '{key}' expects a {expected_type.__name__}."}).encode('utf-8'))
                        return
                        
                    # Verify numeric bounds if applicable
                    if expected_type is int and min_val is not None:
                        if val < min_val or val > max_val:
                            logging.warning(f"Rejection: Client sent out-of-bounds value for key {key}: {val} (allowed range: {min_val}-{max_val})")
                            self.send_response(400)
                            self.send_header('Content-Type', 'application/json')
                            self.send_header('Access-Control-Allow-Origin', '*')
                            self.end_headers()
                            self.wfile.write(json.dumps({"error": f"Bounds Error: Key '{key}' must be between {min_val} and {max_val}."}).encode('utf-8'))
                            return
                
                # 2. Merge validated fields into the existing configuration securely (preserves secrets)
                for key, val in updates.items():
                    current_config[key] = val
                    
                try:
                    save_config(current_config)
                except OSError as e:
                    logging.error(f"Failed to save configuration: {e}")
                    self.send_response(500)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": f"Could not save configuration: {e}"}).encode('utf-8'))
                    return

                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({"status": "success"}).encode('utf-8'))
            except (ValueError, AttributeError) as e:
                logging.warning(f"Rejected malformed config update: {e}")
                self.send_error(400, f"Bad Request: {e}")
        else:
            self.send_error(404, "Not Found")


def run_server():
    # Check for API key and log a warning if missing
    config = load_config()
    if not config.get("API_KEY"):
        logging.warning("=" * 60)
        logging.warning("SECURITY WARNING: No API_KEY is set in config.json!")
        logging.warning("The HTTP API is currently unauthenticated and open to the local network.")
        logging.warning("Anyone can view cameras or overwrite system configuration.")
        logging.warning("=" * 60)
        
    port = 8000
    server_address = ('0.0.0.0', port)
    # Threaded so the dashboard's parallel image fetches don't queue behind each other
    httpd = ThreadingHTTPServer(server_address, KAIRequestHandler)
    logging.info(f"Starting HTTP API server on port {port}...")
    httpd.serve_forever()

if __name__ == "__main__":
    run_server()
