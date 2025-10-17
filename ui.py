import os
from typing import Dict, Tuple

from PIL import Image as PILImage
from kivy.app import App
from kivy.clock import Clock
from kivy.graphics import Color, Line
from kivy.graphics.texture import Texture
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.label import Label

from feeds import append_message, consume_messages

IMAGE_DIR = "images_ready"
FORCE_REQUEST_FEED = "force_request"
FORCE_SERVED_FEED = "force_served"
CONTROL_FEED = "control"
REFRESH_INTERVAL = 30  # seconds


def _parse_ready_filename(filename: str):
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


def discover_cameras() -> Dict[str, Tuple[str, str]]:
    cameras: Dict[str, Tuple[str, str]] = {}
    try:
        for filename in os.listdir(IMAGE_DIR):
            parsed = _parse_ready_filename(filename)
            if parsed:
                camera_id, status = parsed
                cameras[camera_id] = (os.path.join(IMAGE_DIR, filename), status)
    except FileNotFoundError:
        os.makedirs(IMAGE_DIR, exist_ok=True)
    return cameras


class CameraPanel(BoxLayout):
    def __init__(self, camera_id: str, update_cb, toggle_cb, **kwargs):
        super().__init__(orientation="vertical", **kwargs)
        self.camera_id = camera_id
        self.update_cb = update_cb
        self.toggle_cb = toggle_cb
        self.current_status = "UNKNOWN"
        self.power_state = "ON"

        self.image_widget = Image(allow_stretch=True, keep_ratio=True)
        self.add_widget(self.image_widget)

        self.status_label = Label(text="Status: unknown", size_hint_y=None, height=24)
        self.add_widget(self.status_label)

        actions = BoxLayout(size_hint_y=None, height=40)
        force_btn = Button(text="Force Update")
        force_btn.bind(on_press=lambda *_: self.update_cb(self.camera_id))
        actions.add_widget(force_btn)

        toggle_btn = Button(text="Toggle Power")
        toggle_btn.bind(on_press=lambda *_: self._handle_toggle())
        actions.add_widget(toggle_btn)

        self.add_widget(actions)

        with self.canvas.before:
            self._border_color = Color(1, 1, 1, 1)
            self._border = Line(rectangle=(self.x, self.y, self.width, self.height), width=3)
        self.bind(pos=self._update_border, size=self._update_border)

    def _handle_toggle(self):
        new_state = "OFF" if self.power_state == "ON" else "ON"
        self.power_state = new_state
        self.toggle_cb(self.camera_id, new_state)

    def _update_border(self, *_):
        self._border.rectangle = (self.x, self.y, self.width, self.height)

    def refresh(self, image_path: str, status: str) -> None:
        if image_path:
            try:
                pil_image = PILImage.open(image_path).convert("RGB")
                tex = Texture.create(size=pil_image.size)
                tex.blit_buffer(pil_image.tobytes(), colorfmt="rgb", bufferfmt="ubyte")
                tex.flip_vertical()
                self.image_widget.texture = tex
            except Exception as exc:
                print(f"[Dashboard] Failed to load image for {self.camera_id}: {exc}")

        status = status.upper()
        self.current_status = status
        self.status_label.text = f"Status: {status}"
        color = (0, 1, 0, 1) if status == "YES" else (1, 0, 0, 1)
        self._border_color.rgba = color


class Dashboard(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(orientation="vertical", **kwargs)
        self.timer_label = Label(text="Next refresh in: 30s", size_hint_y=None, height=30)
        self.add_widget(self.timer_label)

        self.camera_container = BoxLayout(orientation="horizontal", spacing=10, padding=10)
        self.add_widget(self.camera_container)

        self.elapsed = 0
        self.camera_panels: Dict[str, CameraPanel] = {}
        self.pending_force_updates = set()

        self.load_cameras()
        Clock.schedule_interval(self.update_timer, 1)
        Clock.schedule_interval(self.refresh_all_images, REFRESH_INTERVAL)
        Clock.schedule_interval(self._poll_force_served, 1)

    def load_cameras(self) -> None:
        cameras = discover_cameras()
        # Remove panels for cameras that disappeared
        for cam_id in list(self.camera_panels):
            if cam_id not in cameras:
                panel = self.camera_panels.pop(cam_id)
                self.camera_container.remove_widget(panel)

        # Add missing panels
        for cam_id, (image_path, status) in cameras.items():
            panel = self.camera_panels.get(cam_id)
            if not panel:
                panel = CameraPanel(cam_id, self.request_force_update, self.send_control_command)
                self.camera_panels[cam_id] = panel
                self.camera_container.add_widget(panel)
            panel.refresh(image_path, status)

    def refresh_all_images(self, dt):
        cameras = discover_cameras()
        for cam_id, panel in self.camera_panels.items():
            if cam_id in cameras:
                image_path, status = cameras[cam_id]
                panel.refresh(image_path, status)
        self.elapsed = 0

    def update_timer(self, dt):
        self.elapsed += 1
        remaining = max(0, REFRESH_INTERVAL - self.elapsed)
        self.timer_label.text = f"Next refresh in: {remaining}s"

    def _poll_force_served(self, dt):
        updates = consume_messages(FORCE_SERVED_FEED)
        if not updates:
            return
        cameras = discover_cameras()
        for message in updates:
            if not message.startswith("UPDATED_"):
                continue
            cam_id = message[len("UPDATED_") :]
            self.pending_force_updates.discard(cam_id)
            panel = self.camera_panels.get(cam_id)
            if panel and cam_id in cameras:
                image_path, status = cameras[cam_id]
                panel.refresh(image_path, status)

    def request_force_update(self, camera_id: str) -> None:
        self.pending_force_updates.add(camera_id)
        append_message(FORCE_REQUEST_FEED, f"FORCE_UPDATE_{camera_id}")
        print(f"[Dashboard] Requested force update for {camera_id}")

    def send_control_command(self, camera_id: str, new_state: str) -> None:
        new_state = new_state.upper()
        message = f"SET_{camera_id}_{new_state}"
        append_message(CONTROL_FEED, message)
        print(f"[Dashboard] Sent control command: {message}")


class ControlDashboardApp(App):
    def build(self):
        os.makedirs(IMAGE_DIR, exist_ok=True)
        return Dashboard()


if __name__ == "__main__":
    ControlDashboardApp().run()