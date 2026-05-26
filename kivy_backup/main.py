"""
KAI Dashboard — Android Bootloader Entry Point

This script serves as the main entry point (main.py) required by Buildozer
to package and run the KivyMD Dashboard application on Android devices.
"""

import sys
import os

# Add root and modules directories to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "modules"))

from modules.ui import ControlDashboardApp

if __name__ == "__main__":
    ControlDashboardApp().run()
