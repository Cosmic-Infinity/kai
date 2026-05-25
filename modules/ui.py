"""
KAI Dashboard — IoT Camera Monitoring & Control Interface

A modern, responsive dashboard for monitoring camera feeds and controlling
power states across an organization's rooms.  Uses KivyMD 1.2 Material Design
components with a dark-theme aesthetic.

Functionality:
  • Camera feed grid with live status indicators
  • Force-update per camera (MQTT force_request/force_served)
  • Toggle power per camera (MQTT control feed)
  • Search + status filter + adjustable grid density
  • Fullscreen preview with pan/zoom
  • 30-second auto-refresh cycle
"""

# Disable Kivy multitouch emulation (red dots on middle/right click)
# Must be set before any other Kivy import
import os
os.environ["KIVY_NO_ARGS"] = "1"

from kivy.config import Config
Config.set("input", "mouse", "mouse,multitouch_on_demand")
# Remove Windows touch/pen providers that also create red dots
for provider in ("wm_touch", "wm_pen"):
    Config.remove_option("input", provider)
Config.write()

import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from PIL import Image as PILImage

from kivy.clock import Clock
from kivy.core.window import Window
from kivy.graphics.texture import Texture
from kivy.lang import Builder
from kivy.metrics import dp
from kivy.properties import (
    ListProperty,
    NumericProperty,
    StringProperty,
    BooleanProperty,
    ColorProperty,
)
from kivy.uix.behaviors import ButtonBehavior
from kivy.uix.image import Image, AsyncImage
from kivy.uix.modalview import ModalView
from kivy.utils import get_color_from_hex

from kivymd.app import MDApp
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.button import MDRaisedButton, MDIconButton, MDFlatButton, MDRoundFlatIconButton
from kivymd.uix.screenmanager import MDScreenManager
from kivymd.uix.screen import MDScreen
from kivymd.uix.card import MDCard
from kivy.network.urlrequest import UrlRequest
import requests

def get_client_config_path():
    from kivy.app import App
    app = App.get_running_app()
    if app:
        return os.path.join(app.user_data_dir, "client_config.json")
    return os.path.join(PROJECT_ROOT, "client_config.json")

def load_client_config():
    default_config = {"host": "127.0.0.1", "username": "", "password": "", "theme": "Dark"}
    path = get_client_config_path()
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                loaded = json.load(f)
                for k, v in default_config.items():
                    if k not in loaded:
                        loaded[k] = v
                return loaded
        except:
            pass
    return default_config

def save_client_config(config):
    try:
        path = get_client_config_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(config, f, indent=4)
    except Exception as e:
        print(f"[ClientConfig] Failed to save config: {e}")

class LoginScreen(MDScreen):
    def do_login(self, host, username, password):
        import feeds
        from mqtt_config import MQTT_BROKER_PORT
        feeds.feed_manager.configure(host, MQTT_BROKER_PORT, username, password)
        try:
            r = requests.get(f"http://{host}:8000/api/config", timeout=2)
            if r.status_code == 200:
                feeds.feed_manager.start()
                save_client_config({"host": host, "username": username, "password": password})
                app = MDApp.get_running_app()
                app.dashboard_widget.api_host = host
                server_config = r.json()
                app.dashboard_widget.refresh_interval = server_config.get("DASHBOARD_INTERVAL", 30)
                app.dashboard_widget.elapsed = 0
                app.root.current = "dashboard"
                app.dashboard_widget.load_cameras()
            else:
                print("Invalid HTTP response")
        except Exception as e:
            print(f"Login failed: {e}")

class DashboardScreen(MDScreen):
    pass

class ConfigDialogContent(MDBoxLayout):
    """Custom layout for configuration inputs with interactive info popups."""
    def show_info(self, title, desc):
        from kivymd.uix.snackbar import Snackbar
        Snackbar(
            text=f"[b]{title}:[/b] {desc}",
            bg_color=get_color_from_hex("#1E293B"),
            text_color=get_color_from_hex("#F8FAFC"),
            duration=4,
        ).open()

from kivymd.uix.menu import MDDropdownMenu
from kivymd.uix.dialog import MDDialog
from kivymd.uix.tooltip import MDTooltip

from feeds import append_message, consume_messages
from config_manager import load_config, save_config


# ── Tooltip-enabled widgets (hover on desktop, long-press on mobile) ──

class TooltipRaisedButton(MDRaisedButton, MDTooltip):
    """MDRaisedButton with tooltip support."""
    pass


class TooltipIconButton(MDIconButton, MDTooltip):
    """MDIconButton with tooltip support."""
    pass


class TooltipRoundFlatIconButton(MDRoundFlatIconButton, MDTooltip):
    """MDRoundFlatIconButton with tooltip support."""
    pass


from kivymd.uix.list import OneLineIconListItem

class MenuIconItem(OneLineIconListItem):
    """Custom dropdown menu item that displays an icon and full text without truncation."""
    icon = StringProperty("")

from kivy.factory import Factory
Factory.register('TooltipRaisedButton', cls=TooltipRaisedButton)
Factory.register('TooltipIconButton', cls=TooltipIconButton)
Factory.register('TooltipRoundFlatIconButton', cls=TooltipRoundFlatIconButton)
Factory.register('MenuIconItem', cls=MenuIconItem)

# ─── Paths & constants ────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FORCE_REQUEST_FEED = "force_request"
FORCE_SERVED_FEED = "force_served"
CONTROL_FEED = "control"
REFRESH_INTERVAL = 30  # seconds

# ─── Theme colors ─────────────────────────────────────────────
COLOR_SUCCESS = get_color_from_hex("#10B981")      # emerald — person detected
COLOR_DANGER = get_color_from_hex("#F43F5E")       # rose — no person
COLOR_UNKNOWN = get_color_from_hex("#64748B")      # slate — unknown
COLOR_BADGE_SUCCESS = list(get_color_from_hex("#065F46")) + [0.92] if len(get_color_from_hex("#065F46")) == 3 else get_color_from_hex("#065F46")
COLOR_BADGE_DANGER = list(get_color_from_hex("#9F1239")) + [0.92] if len(get_color_from_hex("#9F1239")) == 3 else get_color_from_hex("#9F1239")
COLOR_BADGE_UNKNOWN = list(get_color_from_hex("#334155")) + [0.92] if len(get_color_from_hex("#334155")) == 3 else get_color_from_hex("#334155")

# Fix badge colors to always be RGBA
def _rgba(hex_color: str, alpha: float = 0.92) -> list:
    """Convert hex to RGBA list."""
    c = get_color_from_hex(hex_color)
    if len(c) == 3:
        return list(c) + [alpha]
    c = list(c)
    c[3] = alpha
    return c

COLOR_BADGE_SUCCESS = _rgba("#065F46")
COLOR_BADGE_DANGER = _rgba("#9F1239")
COLOR_BADGE_UNKNOWN = _rgba("#334155")

POWER_ON_COLOR = get_color_from_hex("#10B981")
POWER_OFF_COLOR = get_color_from_hex("#F43F5E")


_texture_executor = ThreadPoolExecutor(max_workers=4)
_texture_cache = {}

def async_load_texture(image_path: str, callback):
    """Load an image in a background thread (locally or over HTTP) and return pixel data via callback."""
    # Serve directly from cache if available
    if image_path in _texture_cache:
        size, data = _texture_cache[image_path]
        Clock.schedule_once(lambda dt: callback(image_path, size, data), 0)
        return

    def _task():
        try:
            if image_path.startswith("http://") or image_path.startswith("https://"):
                from io import BytesIO
                r = requests.get(image_path, timeout=5)
                if r.status_code == 200:
                    pil_image = PILImage.open(BytesIO(r.content)).convert("RGB")
                else:
                    return
            else:
                if not os.path.exists(image_path):
                    return
                pil_image = PILImage.open(image_path).convert("RGB")
                
            size = pil_image.size
            data = pil_image.tobytes()
            _texture_cache[image_path] = (size, data)
            
            # Prune cache to prevent memory leak (keep latest 50 images)
            if len(_texture_cache) > 50:
                oldest_key = next(iter(_texture_cache))
                del _texture_cache[oldest_key]
                
            Clock.schedule_once(lambda dt: callback(image_path, size, data), 0)
        except Exception as e:
            # Only print if it's not a missing file, as those are expected during cleanup
            if not isinstance(e, FileNotFoundError):
                print(f"[AsyncLoad] Error loading {image_path}: {e}")
    _texture_executor.submit(_task)


# ─── Widgets ──────────────────────────────────────────────────

class ClickableImage(ButtonBehavior, AsyncImage):
    """Image that responds to press/release events and supports parent scrolling."""
    pass


class StatusDot(MDBoxLayout):
    """Tiny coloured dot indicating camera status."""
    dot_color = ListProperty([0.4, 0.5, 0.6, 1])


class CameraCard(MDCard):
    """
    A Material card representing a single camera feed.

    Shows the camera image, detection status, and action buttons.
    All MQTT logic is delegated to the parent ``Dashboard`` via callbacks.
    """

    camera_id = StringProperty("")
    current_status = StringProperty("UNKNOWN")  # YES / NO / UNKNOWN
    current_image_path = StringProperty("")
    ui_state = StringProperty("UNKNOWN")         # ON / OFF / UNKNOWN
    power_state = StringProperty("ON")

    # Derived visual properties (bound in KV)
    status_color = ListProperty(list(COLOR_UNKNOWN))
    status_badge_color = ListProperty(COLOR_BADGE_UNKNOWN)
    status_text = StringProperty("—")
    power_label = StringProperty("ON")
    power_color = ListProperty(list(POWER_ON_COLOR))
    power_badge_color = ListProperty(list(get_color_from_hex("#065F46")))

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            # Manually dispatch touch events to children to prevent MDCard from swallowing the touch
            for child in self.children[:]:
                if child.dispatch('on_touch_down', touch):
                    return True
        return False

    def on_current_status(self, instance, value):
        self.ui_state = "ON" if value == "YES" else "OFF"
        if self.ui_state == "ON":
            self.status_color = list(COLOR_SUCCESS)
            self.status_badge_color = COLOR_BADGE_SUCCESS
            self.status_text = "OCCUPIED"
        else:
            self.status_color = list(COLOR_DANGER)
            self.status_badge_color = COLOR_BADGE_DANGER
            self.status_text = "EMPTY"

    def on_power_state(self, instance, value):
        if value == "ON":
            self.power_label = "ON"
            self.power_color = list(POWER_ON_COLOR)
            self.power_badge_color = list(get_color_from_hex("#065F46"))
        else:
            self.power_label = "OFF"
            self.power_color = list(POWER_OFF_COLOR)
            self.power_badge_color = list(get_color_from_hex("#991B1B"))

    def on_current_image_path(self, instance, value):
        if value:
            self.ids.cam_image.source = value

    # ── callbacks wired from KV ──

    def open_preview(self):
        app = MDApp.get_running_app()
        dashboard = app.dashboard_widget
        if dashboard:
            bboxes = []
            if hasattr(dashboard, 'cameras_data'):
                cam_info = dashboard.cameras_data.get(self.camera_id)
                if cam_info and len(cam_info) > 2:
                    bboxes = cam_info[2]
            print(f"[UI] Opening preview for camera: {self.camera_id} with {len(bboxes)} bounding boxes.")
            dashboard.open_preview(self.camera_id, self.current_image_path, self.current_status, self.power_state, bboxes)

    def on_force_update(self):
        app = MDApp.get_running_app()
        dashboard = app.dashboard_widget
        if dashboard:
            dashboard.request_force_update(self.camera_id)

    def on_toggle_power(self):
        new_state = "OFF" if self.power_state == "ON" else "ON"
        app = MDApp.get_running_app()
        dashboard = app.dashboard_widget
        if dashboard:
            dashboard.send_control_command(self.camera_id, new_state)
            dashboard.set_power_state(self.camera_id, new_state)


class FullscreenPreview(ModalView):
    """
    Full-screen image preview with button-based zoom controls.

    Opened when the user taps a camera image in the grid.
    Uses ScrollView for panning and explicit buttons for zoom.
    """

    camera_id = StringProperty("")
    power_btn_text = StringProperty("Power: ON")
    zoom_text = StringProperty("100%")

    status_color = ListProperty(list(COLOR_UNKNOWN))
    status_badge_color = ListProperty(COLOR_BADGE_UNKNOWN)
    status_text = StringProperty("—")
    power_label = StringProperty("ON")
    power_color = ListProperty(list(POWER_ON_COLOR))
    power_badge_color = ListProperty(list(get_color_from_hex("#065F46")))

    show_bboxes = BooleanProperty(False)

    _zoom_level = 1.0      # current zoom multiplier
    _base_size = (0, 0)    # original image size

    def __init__(self, camera_id: str, image_path: str, status: str,
                 power_state: str, update_cb, toggle_cb, bboxes: list = None, **kwargs):
        self.camera_id = camera_id
        self.image_path = image_path
        self.status = status
        self.power_state = power_state
        self._update_cb = update_cb
        self._toggle_cb = toggle_cb
        self.bboxes = bboxes or []

        self._refresh_labels()
        super().__init__(**kwargs)

        Clock.schedule_once(self._load_and_fit, 0)
        self.bind(on_open=lambda *_: Clock.schedule_once(self.fit_image, 0.05))

    def toggle_bboxes(self):
        print(f"[UI] Toggled bboxes. Current state: {self.show_bboxes} -> {not self.show_bboxes}")
        self.show_bboxes = not self.show_bboxes
        self._update_bboxes_overlay()

    def _update_bboxes_overlay(self):
        img = self.ids.preview_image
        img.canvas.after.clear()
        
        print(f"[UI] _update_bboxes_overlay — show_bboxes: {self.show_bboxes}, bboxes count: {len(self.bboxes)}")
        
        if not self.show_bboxes or not img.texture:
            return
            
        from kivy.graphics import Color, Line, Rectangle
        from kivy.core.text import Label as CoreLabel
        
        # Calculate coordinate boundaries directly using the actual zoomed layout size of the image widget
        draw_w = img.width
        draw_h = img.height
        draw_x = img.x
        draw_y = img.y

        bboxes = self.bboxes

        with img.canvas.after:
            for bbox in bboxes:
                fx, fy, fw, fh, label, color = bbox
                box_color = get_color_from_hex(color)
                
                # Draw the bounding box rectangle
                Color(*box_color)
                x = draw_x + (fx * draw_w)
                y = draw_y + (fy * draw_h)
                w = fw * draw_w
                h = fh * draw_h
                Line(rectangle=(x, y, w, h), width=2)
                
                # Draw the confidence percentage text label badge
                try:
                    # Create and refresh CoreLabel to render text texture
                    core_label = CoreLabel(text=label, font_size=10, bold=True)
                    core_label.options['color'] = (1, 1, 1, 1)
                    core_label.refresh()
                    core_label.refresh()
                    text_texture = core_label.texture
                    text_w, text_h = text_texture.size
                    
                    badge_w = text_w + 8
                    badge_h = text_h + 4
                    badge_x = x
                    badge_y = y + h # Top of bounding box
                    
                    # Ensure the badge does not render off-screen/out-of-image bounds
                    if badge_y + badge_h > draw_y + draw_h:
                        badge_y = y + h - badge_h
                        
                    # 1. Solid semi-transparent dark background for contrast
                    Color(15/255, 23/255, 42/255, 0.85) # #0F172A slate with 85% opacity
                    Rectangle(pos=(badge_x, badge_y), size=(badge_w, badge_h))
                    
                    # 2. Sleek accent border matching the bounding box color
                    Color(*box_color)
                    Line(rectangle=(badge_x, badge_y, badge_w, badge_h), width=1)
                    
                    # 3. Render the text texture
                    Color(1, 1, 1, 1)
                    Rectangle(pos=(badge_x + 4, badge_y + 2), size=(text_w, text_h), texture=text_texture)
                except Exception as text_err:
                    print(f"[UI] Error rendering bbox text label: {text_err}")
                
    def on_size(self, *args):
        if hasattr(self, 'ids') and 'preview_image' in self.ids:
            if self.show_bboxes:
                Clock.schedule_once(lambda dt: self._update_bboxes_overlay(), 0)

    def _refresh_labels(self):
        ui_state = "ON" if self.status == "YES" else "OFF"
        if ui_state == "ON":
            self.status_color = list(COLOR_SUCCESS)
            self.status_badge_color = COLOR_BADGE_SUCCESS
            self.status_text = "OCCUPIED"
        else:
            self.status_color = list(COLOR_DANGER)
            self.status_badge_color = COLOR_BADGE_DANGER
            self.status_text = "EMPTY"

        if self.power_state == "ON":
            self.power_label = "ON"
            self.power_color = list(POWER_ON_COLOR)
            self.power_badge_color = list(get_color_from_hex("#065F46"))
        else:
            self.power_label = "OFF"
            self.power_color = list(POWER_OFF_COLOR)
            self.power_badge_color = list(get_color_from_hex("#991B1B"))

        self.power_btn_text = f"Power: {self.power_state}"

    # ── image loading ──

    def _load_and_fit(self, *_):
        self.reload_image()
        self.fit_image()

    def reload_image(self):
        if self.image_path:
            img = self.ids.preview_image
            img.unbind(texture=self._on_preview_texture_loaded)
            img.bind(texture=self._on_preview_texture_loaded)
            img.source = self.image_path
        self._refresh_labels()

    def _on_preview_texture_loaded(self, img, texture):
        if texture:
            self._base_size = texture.size
            self.fit_image()

    def fit_image(self, *_):
        """Reset zoom to fit the image within the scroll area."""
        img = self.ids.preview_image
        if not img.texture:
            return
        scroll = self.ids.preview_scroll
        sw, sh = scroll.size
        bw, bh = self._base_size
        if bw <= 0 or bh <= 0 or sw <= 0 or sh <= 0:
            return
        self._zoom_level = min(sw / bw, sh / bh)
        self._apply_zoom()

    def zoom_in(self, *_):
        """Increase zoom by 25%."""
        self._zoom_level = min(self._zoom_level * 1.25, 10.0)
        self._apply_zoom()

    def zoom_out(self, *_):
        """Decrease zoom by 25%."""
        self._zoom_level = max(self._zoom_level * 0.8, 0.1)
        self._apply_zoom()

    def _apply_zoom(self):
        """Apply current zoom level to the image size."""
        bw, bh = self._base_size
        if bw <= 0 or bh <= 0:
            return
        img = self.ids.preview_image
        img.size = (bw * self._zoom_level, bh * self._zoom_level)
        pct = int(self._zoom_level * 100)
        self.zoom_text = f"{pct}%"

        # Explicitly trigger canvas redraw after modifying zoom boundary size on the screen layout bounds
        if self.show_bboxes:
            Clock.schedule_once(lambda dt: self._update_bboxes_overlay(), 0)

    # ── actions ──

    def do_force_update(self):
        self._update_cb(self.camera_id)

    def do_toggle_power(self):
        self.power_state = "OFF" if self.power_state == "ON" else "ON"
        self._refresh_labels()
        self._toggle_cb(self.camera_id, self.power_state)

    def close_preview(self, *_):
        """Explicitly dismiss the preview modal."""
        self.dismiss()


class Dashboard(MDBoxLayout):
    """
    Main dashboard widget.

    Discovers cameras, builds the card grid, handles MQTT messaging,
    and drives the 30-second refresh cycle.
    """

    grid_icon = StringProperty("view-grid")
    refresh_interval = NumericProperty(REFRESH_INTERVAL)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        config = load_config()
        self.refresh_interval = config.get("DASHBOARD_INTERVAL", 30)
        
        self.elapsed = 0
        self.pending_force_updates: set = set()
        self.filter_menu = None
        self.current_filter = "All"
        self.camera_power_states: Dict[str, str] = {}
        self.cameras_data: Dict[str, Tuple[str, str]] = {}

        # Wait one frame for KV ids to be available
        Clock.schedule_once(self._post_init, 0)

    def _post_init(self, *_):
        """Called after KV tree is built so all ids are available."""
        Window.bind(size=self._on_window_resize)
        self._auto_density()

        self.load_cameras()

        Clock.schedule_interval(self.update_timer, 1)
        Clock.schedule_interval(self._poll_force_served, 1)

    # ─── camera discovery ─────────────────────────────────────

    def load_cameras(self) -> None:
        if not hasattr(self, 'api_host') or not self.api_host:
            return
            
        try:
            r = requests.get(f"http://{self.api_host}:8000/api/cameras", timeout=5)
            if r.status_code == 200:
                data = r.json()
                self.cameras_data = {}
                timestamp = int(time.time())
                for cam_id, info in data.items():
                    # Construct full HTTP URL for the image
                    img_url = f"http://{self.api_host}:8000{info['image_path']}?ts={timestamp}"
                    self.cameras_data[cam_id] = (img_url, info["status"], info.get("bboxes", []))
                self.apply_filters_and_layout()
        except Exception as e:
            print(f"Failed to fetch cameras: {e}")

    # ─── grid / filters ──────────────────────────────────────

    def apply_filters_and_layout(self):
        rv_data = []
        for cam_id, cam_info in sorted(self.cameras_data.items()):
            image_path = cam_info[0]
            status = cam_info[1]
            ui_state = "ON" if status.upper() == "YES" else "OFF"
            
            # check filter
            search = self.ids.search_field.text.strip().lower()
            if search and search not in cam_id.lower():
                continue
                
            power_state = self.camera_power_states.get(cam_id, "ON")
            
            # evaluate combinations
            f = self.current_filter
            if "Occupied" in f and ui_state != "ON":
                continue
            if "Empty" in f and ui_state != "OFF":
                continue
            if "Power ON" in f and power_state != "ON":
                continue
            if "Power OFF" in f and power_state != "OFF":
                continue

            rv_data.append({
                "camera_id": cam_id,
                "current_image_path": image_path,
                "current_status": status.upper(),
                "power_state": power_state,
            })

        self.ids.scroll_area.data = rv_data
        
        # Handle initial load where grid density was unapplied if the slider was missing
        if getattr(self.ids.camera_grid, 'cols', None) is None or self.ids.camera_grid.cols == 0:
            self._apply_density(3)
            
        self.ids.camera_count_label.text = f"{len(rv_data)} camera{'s' if len(rv_data) != 1 else ''}"

    def set_power_state(self, camera_id, state):
        self.camera_power_states[camera_id] = state
        self.apply_filters_and_layout()

    def open_filter_menu(self):
        if hasattr(self, 'filter_menu') and self.filter_menu:
            menu = self.filter_menu
            Clock.schedule_once(lambda dt: menu.dismiss(), 0)
            self.filter_menu = None
            return

        app = MDApp.get_running_app()
        if not self.filter_menu:
            filters = [
                "All", "Occupied", "Empty", 
                "Power ON", "Power OFF",
                "Occupied & Power ON", "Occupied & Power OFF",
                "Empty & Power ON", "Empty & Power OFF"
            ]
            filter_icons = {
                "All": "tag-multiple-outline",
                "Occupied": "account-check",
                "Empty": "account-off",
                "Power ON": "power",
                "Power OFF": "power-off",
                "Occupied & Power ON": "flash",
                "Occupied & Power OFF": "flash-off",
                "Empty & Power ON": "lightning-bolt-outline",
                "Empty & Power OFF": "power-sleep"
            }
            menu_items = [
                {
                    "text": item,
                    "viewclass": "MenuIconItem",
                    "icon": filter_icons.get(item, "tag-outline"),
                    "on_release": lambda *a, x=item: self.set_filter(x),
                } for item in filters
            ]
            self.filter_menu = MDDropdownMenu(
                caller=self.ids.filter_btn,
                items=menu_items,
                width_mult=7 if Window.width > dp(500) else 5,
            )
            self.filter_menu.bind(on_dismiss=lambda *a: setattr(self, 'filter_menu', None))
        Clock.schedule_once(lambda dt: self.filter_menu.open(), 0)

    def open_settings_menu(self, caller_button):
        if hasattr(self, 'settings_menu') and self.settings_menu:
            menu = self.settings_menu
            Clock.schedule_once(lambda dt: menu.dismiss(), 0)
            self.settings_menu = None
            return

        app = MDApp.get_running_app()
        theme_text = "Switch to Light Mode" if app.theme_cls.theme_style == "Dark" else "Switch to Dark Mode"
        
        menu_items = [
            {
                "text": theme_text,
                "viewclass": "MenuIconItem",
                "icon": "theme-light-dark",
                "on_release": self.toggle_theme_style,
            },
            {
                "text": "Refresh Timers",
                "viewclass": "MenuIconItem",
                "icon": "cog-outline",
                "on_release": self.open_settings_dialog,
            },
            {
                "text": "Switch Server / Logout",
                "viewclass": "MenuIconItem",
                "icon": "logout",
                "on_release": self.logout,
            }
        ]
        self.settings_menu = MDDropdownMenu(
            caller=caller_button,
            items=menu_items,
            width_mult=6.5 if Window.width > dp(500) else 4.5,
        )
        self.settings_menu.bind(on_dismiss=lambda *a: setattr(self, 'settings_menu', None))
        Clock.schedule_once(lambda dt: self.settings_menu.open(), 0)

    def toggle_theme_style(self, *_):
        if self.settings_menu:
            menu = self.settings_menu
            Clock.schedule_once(lambda dt: menu.dismiss(), 0)
        app = MDApp.get_running_app()
        app.toggle_theme()

    def logout(self, *_):
        if hasattr(self, 'settings_menu') and self.settings_menu:
            menu = self.settings_menu
            Clock.schedule_once(lambda dt: menu.dismiss(), 0)
        save_client_config({"host": "", "username": "", "password": ""})
        app = MDApp.get_running_app()
        app.root.current = "login"

    def open_settings_dialog(self, *_):
        if hasattr(self, 'settings_menu') and self.settings_menu:
            menu = self.settings_menu
            Clock.schedule_once(lambda dt: menu.dismiss(), 0)
        if not hasattr(self, 'settings_dialog'):
            from kivy.factory import Factory
            self.dialog_content = Factory.ConfigDialogContent()
            self.settings_dialog = MDDialog(
                title="System Configuration",
                type="custom",
                content_cls=self.dialog_content,
                buttons=[
                    MDFlatButton(
                        text="CANCEL",
                        theme_text_color="Custom",
                        text_color=self.theme_cls.primary_color,
                        on_release=lambda x: self.settings_dialog.dismiss()
                    ),
                    MDFlatButton(
                        text="SAVE",
                        theme_text_color="Custom",
                        text_color=self.theme_cls.primary_color,
                        on_release=self.save_settings
                    ),
                ],
            )
        
        # Load current values into dialog via HTTP
        try:
            r = requests.get(f"http://{self.api_host}:8000/api/config", timeout=2)
            if r.status_code == 200:
                config = r.json()
                self.dialog_content.ids.image_interval.text = str(config.get("IMAGE_SERVER_INTERVAL", 60))
                self.dialog_content.ids.dash_interval.text = str(config.get("DASHBOARD_INTERVAL", 30))
                self.dialog_content.ids.control_interval.text = str(config.get("CONTROL_SERVER_INTERVAL", 30))
                self.dialog_content.ids.threshold.text = str(config.get("INACTIVITY_THRESHOLD", 10))
                self.dialog_content.ids.mqtt_keepalive.text = str(config.get("MQTT_KEEPALIVE", 120))
                self.dialog_content.ids.mqtt_reconnect_delay.text = str(config.get("MQTT_RECONNECT_DELAY", 2))
        except Exception as e:
            print(f"Failed to load config: {e}")
        
        self.settings_dialog.open()

    def save_settings(self, *_):
        try:
            new_config = {
                "IMAGE_SERVER_INTERVAL": int(self.dialog_content.ids.image_interval.text or 60),
                "DASHBOARD_INTERVAL": int(self.dialog_content.ids.dash_interval.text or 30),
                "CONTROL_SERVER_INTERVAL": int(self.dialog_content.ids.control_interval.text or 30),
                "INACTIVITY_THRESHOLD": int(self.dialog_content.ids.threshold.text or 10),
                "MQTT_KEEPALIVE": int(self.dialog_content.ids.mqtt_keepalive.text or 120),
                "MQTT_RECONNECT_DELAY": int(self.dialog_content.ids.mqtt_reconnect_delay.text or 2),
            }
            # Post config via HTTP API
            try:
                requests.post(f"http://{self.api_host}:8000/api/config", json=new_config, timeout=2)
            except Exception as e:
                print(f"Failed to post config: {e}")
            
            # Apply locally immediately
            self.refresh_interval = new_config["DASHBOARD_INTERVAL"]
            self.elapsed = 0
            self.ids.timer_label.text = f"{self.refresh_interval}s"
            
            self.settings_dialog.dismiss()
        except ValueError:
            print("[UI] Invalid configuration values entered.")

    def set_filter(self, status_filter):
        self.current_filter = status_filter
        self.ids.filter_btn.text = f"Filter: {status_filter}"
        if self.filter_menu:
            menu = self.filter_menu
            Clock.schedule_once(lambda dt: menu.dismiss(), 0)
        self.on_filter_changed()

    def _apply_density(self, value):
        cols = max(1, int(round(value)))
        self.ids.camera_grid.cols = cols
        
        # Determine the grid icon based on the number of columns
        if cols == 1:
            self.grid_icon = "view-stream"
        elif cols == 2:
            self.grid_icon = "view-grid-outline"
        elif cols >= 3:
            self.grid_icon = "view-grid"
        elif cols >= 4:
            self.grid_icon = "view-comfy"

    def on_filter_changed(self, *_):
        self.apply_filters_and_layout()

    def cycle_density(self):
        """Cycle grid density when toolbar icon is pressed."""
        cols = self.ids.camera_grid.cols
        new_val = (cols % 5) + 1
        self._apply_density(new_val)

    def manual_refresh(self):
        """Force an immediate refresh when toolbar icon is pressed."""
        self.refresh_all_images(0)

    # ─── responsive density ───────────────────────────────────

    def _on_window_resize(self, *_):
        self._auto_density()

    def _auto_density(self):
        """Set grid density based on window width for responsive behaviour."""
        w = Window.width
        if w < dp(500):
            cols = 1
        elif w < dp(800):
            cols = 2
        elif w < dp(1100):
            cols = 3
        else:
            cols = 4
        self._apply_density(cols)

    # ─── refresh cycle ────────────────────────────────────────

    def update_timer(self, dt):
        self.elapsed += 1
        remaining = max(0, self.refresh_interval - self.elapsed)
        self.ids.timer_label.text = f"{remaining}s"
        
        if remaining == 0:
            self.refresh_all_images(0)

    def refresh_all_images(self, dt):
        self.load_cameras()
        self.elapsed = 0

    # ─── MQTT: force update ───────────────────────────────────

    def _poll_force_served(self, dt):
        messages = consume_messages(FORCE_SERVED_FEED)
        for msg in messages:
            if msg.startswith("FORCE_SERVED_"):
                cam_id = msg.replace("FORCE_SERVED_", "")
                if cam_id in self.pending_force_updates:
                    self.pending_force_updates.remove(cam_id)
                    print(f"[Dashboard] Force update completed for {cam_id}")
        
        if messages:
            self.load_cameras()

    def request_force_update(self, camera_id: str) -> None:
        self.pending_force_updates.add(camera_id)
        append_message(FORCE_REQUEST_FEED, f"FORCE_UPDATE_{camera_id}")
        print(f"[Dashboard] Requested force update for {camera_id}")

    # ─── MQTT: control ────────────────────────────────────────

    def send_control_command(self, camera_id: str, new_state: str) -> None:
        new_state = new_state.upper()
        message = f"SET_{camera_id}_{new_state}"
        append_message(CONTROL_FEED, message)
        print(f"[Dashboard] Sent control command: {message}")

    def set_power_state(self, camera_id: str, new_state: str) -> None:
        self.camera_power_states[camera_id] = new_state
        self.apply_filters_and_layout()

    # ─── preview ──────────────────────────────────────────────

    def open_preview(self, camera_id: str, image_path: str,
                     status: str, power_state: str, bboxes: list = None) -> None:
        preview = FullscreenPreview(
            camera_id, image_path, status, power_state,
            self.request_force_update, self.send_control_command, bboxes or []
        )
        preview.open()


# ─── App ──────────────────────────────────────────────────────

class WindowManager(MDScreenManager):
    pass

class ControlDashboardApp(MDApp):
    # Dynamic theme color properties that automatically bind in KV
    bg_primary = ColorProperty(get_color_from_hex("#0F172A"))
    bg_card = ColorProperty(get_color_from_hex("#1E293B"))
    text_primary = ColorProperty(get_color_from_hex("#F8FAFC"))
    text_secondary = ColorProperty(get_color_from_hex("#94A3B8"))
    border_color = ColorProperty(get_color_from_hex("#334155"))
    elevation_card = NumericProperty(6)
    elevation_dialog = NumericProperty(4)
    status_bar_height = NumericProperty(0)

    def apply_theme_colors(self, theme_style):
        if theme_style == "Light":
            self.bg_primary = get_color_from_hex("#F1F5F9")      # slate-100 (light bg)
            self.bg_card = get_color_from_hex("#FFFFFF")         # pure white card bg
            self.text_primary = get_color_from_hex("#0F172A")     # slate-900 (deep dark text)
            self.text_secondary = get_color_from_hex("#475569")   # slate-600 (muted text)
            self.border_color = get_color_from_hex("#CBD5E1")     # slate-300 (light border)
            self.elevation_card = 2                              # subtle card shadow
            self.elevation_dialog = 1                            # subtle header shadow
        else:
            self.bg_primary = get_color_from_hex("#0F172A")      # deep navy bg
            self.bg_card = get_color_from_hex("#1E293B")         # slate card bg
            self.text_primary = get_color_from_hex("#F8FAFC")     # near-white text
            self.text_secondary = get_color_from_hex("#94A3B8")   # muted slate text
            self.border_color = get_color_from_hex("#334155")     # slate border
            self.elevation_card = 6
            self.elevation_dialog = 4

    def toggle_theme(self):
        self.theme_cls.theme_style = "Light" if self.theme_cls.theme_style == "Dark" else "Dark"
        cfg = load_client_config()
        cfg["theme"] = self.theme_cls.theme_style
        save_client_config(cfg)
        self.apply_theme_colors(self.theme_cls.theme_style)

    def build(self):
        self.title = "kai Dashboard"
        from kivy.core.window import Window
        Window.fullscreen = False
        self.status_bar_height = self.get_android_status_bar_height()
        
        # Force show Android status bar and disable fullscreen
        from kivy.utils import platform
        if platform == 'android':
            try:
                from jnius import autoclass
                PythonActivity = autoclass('org.kivy.android.PythonActivity')
                activity = PythonActivity.mActivity
                AndroidWindowManager = autoclass('android.view.WindowManager$LayoutParams')
                
                from android.runnable import run_on_ui_thread
                
                @run_on_ui_thread
                def show_status_bar():
                    window = activity.getWindow()
                    window.clearFlags(AndroidWindowManager.FLAG_FULLSCREEN)
                    window.addFlags(AndroidWindowManager.FLAG_FORCE_NOT_FULLSCREEN)
                
                show_status_bar()
            except Exception as e:
                print(f"[AndroidFullscreen] Failed to show status bar: {e}")
        
        client_cfg = load_client_config()
        theme = client_cfg.get("theme", "Dark")
        self.theme_cls.theme_style = theme
        self.apply_theme_colors(theme)

        self.theme_cls.primary_palette = "Blue"
        self.theme_cls.accent_palette = "Teal"
        self.theme_cls.material_style = "M3"

        kv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui.kv")
        Builder.load_file(kv_path)
        
        sm = WindowManager()
        self.login_screen = LoginScreen(name="login")
        self.dashboard_screen = DashboardScreen(name="dashboard")
        
        # Add dashboard inside dashboard screen
        self.dashboard_widget = Dashboard()
        self.dashboard_screen.add_widget(self.dashboard_widget)
        
        sm.add_widget(self.login_screen)
        sm.add_widget(self.dashboard_screen)
        
        # Auto login if cached
        client_cfg = load_client_config()
        if client_cfg.get("host") and client_cfg.get("username"):
            # Set credentials and test
            import feeds
            from mqtt_config import MQTT_BROKER_PORT
            feeds.feed_manager.configure(client_cfg["host"], MQTT_BROKER_PORT, client_cfg["username"], client_cfg["password"])
            try:
                feeds.feed_manager.start()
                # Also test HTTP API
                r = requests.get(f"http://{client_cfg['host']}:8000/api/config", timeout=2)
                if r.status_code == 200:
                    self.dashboard_widget.api_host = client_cfg["host"]
                    sm.current = "dashboard"
            except Exception as e:
                print(f"Auto-login failed: {e}")
                sm.current = "login"
        
        return sm

    def get_android_status_bar_height(self) -> float:
        from kivy.utils import platform
        if platform == 'android':
            try:
                from jnius import autoclass
                PythonActivity = autoclass('org.kivy.android.PythonActivity')
                activity = PythonActivity.mActivity
                resources = activity.getResources()
                resource_id = resources.getIdentifier('status_bar_height', 'dimen', 'android')
                if resource_id > 0:
                    height_px = resources.getDimensionPixelSize(resource_id)
                    return float(height_px / dp(1))
            except Exception as e:
                print(f"[AndroidStatusBar] Failed to retrieve height: {e}")
        return 0.0

if __name__ == "__main__":
    ControlDashboardApp().run()