import os
import sys
import json
import logging
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse

# Setup path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_manager import load_config

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_DIR = os.path.join(PROJECT_ROOT, "images_ready")

def discover_cameras():
    cameras = {}
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
            # Attempt to load bboxes
            bbox_path = os.path.join(IMAGE_DIR, f"{camera_id}_bboxes.json")
            if os.path.exists(bbox_path):
                try:
                    with open(bbox_path, 'r') as f:
                        cameras[camera_id]["bboxes"] = json.load(f)
                except: pass
    except FileNotFoundError:
        pass
    return cameras

class KAIRequestHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
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
            self.wfile.write(json.dumps(load_config()).encode('utf-8'))
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
        parsed = urlparse(self.path)
        if parsed.path == "/api/config":
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            try:
                new_config = json.loads(post_data.decode('utf-8'))
                from config_manager import save_config
                save_config(new_config)
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({"status": "success"}).encode('utf-8'))
            except Exception as e:
                self.send_error(400, f"Bad Request: {e}")
        else:
            self.send_error(404, "Not Found")

def run_server():
    port = 8000
    server_address = ('0.0.0.0', port)
    httpd = HTTPServer(server_address, KAIRequestHandler)
    logging.info(f"Starting HTTP API server on port {port}...")
    httpd.serve_forever()

if __name__ == "__main__":
    run_server()
