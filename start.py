"""
KAI IoT System Launcher

This script launches all system modules and the MQTT monitor in separate terminal windows.
It provides a convenient single-command startup for the entire system.
"""

import os
import sys
import platform
import subprocess
import time
from pathlib import Path


# Color codes for terminal output
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


def print_banner():
    """Print the startup banner."""
    print(colored("=" * 70, Colors.CYAN))
    print(colored("[kai] IoT System Launcher", Colors.HEADER + Colors.BOLD))
    print(colored("=" * 70, Colors.CYAN))
    print()


def check_python():
    """Verify Python is available."""
    try:
        version = sys.version.split()[0]
        print(colored(f"[OK] Python {version}", Colors.GREEN))
        return True
    except Exception as e:
        print(colored(f"[FAIL] Python check failed: {e}", Colors.RED))
        return False


def check_mqtt_broker():
    """Check if MQTT broker is running."""
    try:
        import paho.mqtt.client as mqtt
        
        connected = [False]
        
        def on_connect(client, userdata, flags, rc):
            connected[0] = (rc == 0)
            client.disconnect()
        
        client = mqtt.Client(client_id="kai_launcher_test")
        client.on_connect = on_connect
        client.connect("localhost", 1883, 60)
        client.loop_start()
        time.sleep(1)
        client.loop_stop()
        
        if connected[0]:
            print(colored("[OK] MQTT broker is running", Colors.GREEN))
            return True
        else:
            print(colored("[FAIL] MQTT broker connection failed", Colors.RED))
            print(colored("  Run: net start mosquitto", Colors.YELLOW))
            return False
            
    except ImportError:
        print(colored("[FAIL] paho-mqtt not installed", Colors.RED))
        print(colored("  Run: pip install paho-mqtt", Colors.YELLOW))
        return False
    except Exception as e:
        print(colored(f"[FAIL] MQTT broker not reachable: {e}", Colors.RED))
        print(colored("  Run: net start mosquitto", Colors.YELLOW))
        return False


def check_directories():
    """Verify required directories exist."""
    dirs = {
        "modules": Path("modules"),
        "images_src": Path("images_src"),
        "images_ready": Path("images_ready"),
    }
    
    all_exist = True
    for name, path in dirs.items():
        if path.exists():
            print(colored(f"[OK] {name}/ directory exists", Colors.GREEN))
        else:
            print(colored(f"[FAIL] {name}/ directory missing", Colors.RED))
            all_exist = False
            # Create if missing
            path.mkdir(exist_ok=True)
            print(colored(f"  Created {name}/ directory", Colors.YELLOW))
    
    return all_exist


def launch_windows(modules):
    """Launch modules in separate PowerShell windows on Windows."""
    processes = []
    
    for module in modules:
        name, script, title = module['name'], module['script'], module['title']
        
        try:
            # PowerShell command to run the script with a custom title
            cmd = [
                'powershell.exe',
                '-NoExit',
                '-Command',
                f'$host.UI.RawUI.WindowTitle = "{title}"; python {script}'
            ]
            
            # Start new PowerShell window
            process = subprocess.Popen(
                cmd,
                creationflags=subprocess.CREATE_NEW_CONSOLE,
                cwd=os.getcwd()
            )
            processes.append(process)
            print(colored(f"[OK] Launched {name}", Colors.GREEN))
            time.sleep(0.5)  # Stagger launches
            
        except Exception as e:
            print(colored(f"[FAIL] Failed to launch {name}: {e}", Colors.RED))
    
    return processes


def launch_unix(modules):
    """Launch modules in separate terminal windows on Linux/Mac."""
    processes = []
    
    # Detect available terminal emulator
    terminals = [
        ('gnome-terminal', ['gnome-terminal', '--', 'bash', '-c']),
        ('xterm', ['xterm', '-hold', '-e']),
        ('konsole', ['konsole', '-e']),
        ('Terminal', ['open', '-a', 'Terminal']),  # macOS
    ]
    
    terminal_cmd = None
    for term_name, term_cmd in terminals:
        try:
            subprocess.run(['which', term_name], capture_output=True, check=True)
            terminal_cmd = term_cmd
            break
        except:
            continue
    
    if not terminal_cmd:
        print(colored("[FAIL] No suitable terminal emulator found", Colors.RED))
        return processes
    
    for module in modules:
        name, script, title = module['name'], module['script'], module['title']
        
        try:
            cmd = terminal_cmd + [f'python {script}; exec bash']
            process = subprocess.Popen(cmd, cwd=os.getcwd())
            processes.append(process)
            print(colored(f"[OK] Launched {name}", Colors.GREEN))
            time.sleep(0.5)
            
        except Exception as e:
            print(colored(f"[FAIL] Failed to launch {name}: {e}", Colors.RED))
    
    return processes


def main():
    """Main launcher function."""
    print_banner()
    
    # Pre-flight checks
    print(colored("Running pre-flight checks...", Colors.CYAN))
    print()
    
    checks_passed = True
    checks_passed &= check_python()
    checks_passed &= check_directories()
    checks_passed &= check_mqtt_broker()
    
    print()
    
    if not checks_passed:
        print(colored("[WARN] Some checks failed. Continue anyway? (y/N): ", Colors.YELLOW), end='')
        response = input().strip().lower()
        if response != 'y':
            print(colored("Startup cancelled.", Colors.RED))
            return
        print()
    
    # Define modules to launch
    modules = [
        {
            'name': 'Image Server',
            'script': 'modules\\image_server.py' if platform.system() == 'Windows' else 'modules/image_server.py',
            'title': 'kai - Image Server'
        },
        {
            'name': 'Control Server',
            'script': 'modules\\control_server.py' if platform.system() == 'Windows' else 'modules/control_server.py',
            'title': 'kai - Control Server'
        },
        {
            'name': 'MQTT Monitor',
            'script': 'tools\\monitor_mqtt.py' if platform.system() == 'Windows' else 'tools/monitor_mqtt.py',
            'title': 'kai - MQTT Monitor'
        },
        {
            'name': 'HTTP API Server',
            'script': 'modules\\http_api.py' if platform.system() == 'Windows' else 'modules/http_api.py',
            'title': 'kai - HTTP API'
        },
        {
            'name': 'Dashboard UI',
            'script': 'modules\\ui.py' if platform.system() == 'Windows' else 'modules/ui.py',
            'title': 'kai - Dashboard Terminal'
        },
    ]
    
    # Launch modules
    print(colored("Launching modules...", Colors.CYAN))
    print()
    
    if platform.system() == 'Windows':
        processes = launch_windows(modules)
    else:
        processes = launch_unix(modules)
    
    print()
    print(colored("=" * 70, Colors.CYAN))
    print(colored(f"[OK] Launched {len(processes)} modules successfully!", Colors.GREEN + Colors.BOLD))
    print(colored("=" * 70, Colors.CYAN))
    print()
    print(colored("System is starting up...", Colors.CYAN))
    print()
    print(colored("Modules launched:", Colors.BOLD))
    for module in modules:
        print(f"  • {module['name']}")
    print()
    print(colored("To stop the system:", Colors.YELLOW))
    print("  1. Close each terminal window, OR")
    print("  2. Press Ctrl+C in each window")
    print()
    print(colored("Tip: Check the MQTT Monitor window to see real-time message traffic!", Colors.CYAN))
    print()
    
    # Keep this script running
    try:
        print(colored("Press Ctrl+C here to exit this launcher (modules will continue running)", Colors.YELLOW))
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print(colored("\n\nLauncher stopped. Modules are still running.", Colors.YELLOW))
        print(colored("Close individual terminal windows to stop each module.", Colors.YELLOW))


if __name__ == "__main__":
    main()
