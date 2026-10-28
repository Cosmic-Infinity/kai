"""
kai IoT System Launcher

This script launches all system modules and the MQTT monitor in separate terminal windows.
It provides a convenient single-command startup for the entire system.
"""

import os
import sys
import platform
import shlex
import shutil
import subprocess
import time
from pathlib import Path

# Always resolve paths from the project root, no matter where the launcher is started from
PROJECT_ROOT = Path(__file__).resolve().parent
LOG_DIR = PROJECT_ROOT / "logs"

# Never crash on a console that can't encode the banner/box characters (e.g. redirected output)
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except AttributeError:
        pass


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


def broker_start_hint():
    """Return the OS-appropriate command to start the Mosquitto service."""
    system = platform.system()
    if system == 'Windows':
        return "net start mosquitto (Administrator shell)"
    if system == 'Darwin':
        return "brew services start mosquitto"
    return "sudo systemctl start mosquitto"


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
        
        from modules.config_manager import get_config_namespace
        config = get_config_namespace()
        
        connected = [False]
        
        def on_connect(client, userdata, flags, *args, **kwargs):
            rc = args[0] if args else kwargs.get('reason_code', kwargs.get('rc', 0))
            connected[0] = (rc == 0)
            client.disconnect()
        
        try:
            client = mqtt.Client(
                client_id="kai_launcher_test",
                callback_api_version=mqtt.CallbackAPIVersion.VERSION2
            )
        except (AttributeError, TypeError):
            client = mqtt.Client(client_id="kai_launcher_test")
            
        client.on_connect = on_connect
        
        if config.MQTT_USERNAME and config.MQTT_PASSWORD:
            client.username_pw_set(config.MQTT_USERNAME, config.MQTT_PASSWORD)
            
        client.connect(config.MQTT_BROKER_HOST, config.MQTT_BROKER_PORT, 60)
        client.loop_start()
        time.sleep(1)
        client.loop_stop()
        
        if connected[0]:
            print(colored("[OK] MQTT broker is running", Colors.GREEN))
            return True
        else:
            print(colored("[FAIL] MQTT broker connection failed (could not authenticate or connect)", Colors.RED))
            print(colored("  Verify your credentials in config/config.json are correct.", Colors.YELLOW))
            return False
            
    except ImportError:
        print(colored("[FAIL] paho-mqtt not installed", Colors.RED))
        print(colored("  Run: pip install paho-mqtt", Colors.YELLOW))
        return False
    except Exception as e:
        print(colored(f"[FAIL] MQTT broker not reachable: {e}", Colors.RED))
        print(colored(f"  Start the broker with: {broker_start_hint()}", Colors.YELLOW))
        return False


def check_sample_credentials():
    """Warn (without failing the pre-flight) while config.json still carries the shipped demo credentials."""
    from modules.config_manager import load_config
    notice = load_config().get("_NOTICE")
    if notice:
        print(colored("[WARN] config/config.json still contains the SAMPLE credentials shipped with the repository.", Colors.YELLOW))
        print(colored("  Run `python configure_kai.py` to generate your own before a real deployment.", Colors.YELLOW))


def check_directories():
    """Verify required directories exist."""
    dirs = {
        "modules": PROJECT_ROOT / "modules",
        "images_src": PROJECT_ROOT / "images_src",
        "images_ready": PROJECT_ROOT / "images_ready",
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


def _ps_quote(text):
    """Quote a string for a PowerShell single-quoted literal."""
    return "'" + str(text).replace("'", "''") + "'"


def _shell_command(module):
    """POSIX shell command line that runs a module from the project root."""
    return (
        f"printf '\\033]0;%s\\007' {shlex.quote(module['title'])}; "
        f"cd {shlex.quote(str(PROJECT_ROOT))} && "
        f"{shlex.quote(sys.executable)} {shlex.quote(module['script'])}"
    )


def _applescript_quote(text):
    return '"' + text.replace('\\', '\\\\').replace('"', '\\"') + '"'


def _linux_terminal_argv(module):
    """Build argv that opens a terminal window running the module, or None if no terminal/display is available."""
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return None
    shell_cmd = _shell_command(module) + "; exec bash"
    title = module['title']
    candidates = [
        ('gnome-terminal', ['gnome-terminal', f'--title={title}', '--', 'bash', '-c', shell_cmd]),
        ('konsole', ['konsole', '-e', 'bash', '-c', shell_cmd]),
        ('xfce4-terminal', ['xfce4-terminal', '--title', title, '-x', 'bash', '-c', shell_cmd]),
        ('xterm', ['xterm', '-T', title, '-hold', '-e', 'bash', '-c', shell_cmd]),
    ]
    for exe, argv in candidates:
        if shutil.which(exe):
            return argv
    return None


def launch_background(module):
    """Run a module detached with its output in logs/<name>.log (no terminal available, e.g. headless server)."""
    LOG_DIR.mkdir(exist_ok=True)
    log_name = module['script'].replace(os.sep, '_').replace('.py', '') + '.log'
    log_path = LOG_DIR / log_name
    with open(log_path, 'ab') as log_file:
        process = subprocess.Popen(
            [sys.executable, module['script']],
            cwd=PROJECT_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=(os.name != 'nt'),
        )
    print(colored(f"  Output: {log_path} (PID {process.pid})", Colors.CYAN))
    return process


def launch_module(module):
    """Open a module in its own terminal window using whatever the current OS provides."""
    system = platform.system()

    if system == 'Windows':
        command = (
            f"$host.UI.RawUI.WindowTitle = {_ps_quote(module['title'])}; "
            f"& {_ps_quote(sys.executable)} {_ps_quote(module['script'])}"
        )
        return subprocess.Popen(
            ['powershell.exe', '-NoExit', '-Command', command],
            creationflags=subprocess.CREATE_NEW_CONSOLE,
            cwd=PROJECT_ROOT,
        )

    if system == 'Darwin':
        script = f'tell application "Terminal" to do script {_applescript_quote(_shell_command(module))}'
        return subprocess.Popen(['osascript', '-e', script, '-e', 'tell application "Terminal" to activate'])

    argv = _linux_terminal_argv(module)
    if argv:
        return subprocess.Popen(argv, cwd=PROJECT_ROOT)

    print(colored("  No graphical terminal found - running in the background instead.", Colors.YELLOW))
    return launch_background(module)


def launch_modules(modules):
    """Launch every module, reporting (not hiding) any that fail to start."""
    processes = []
    for module in modules:
        try:
            processes.append(launch_module(module))
            print(colored(f"[OK] Launched {module['name']}", Colors.GREEN))
            time.sleep(0.5)  # Stagger launches
        except OSError as e:
            print(colored(f"[FAIL] Failed to launch {module['name']}: {e}", Colors.RED))
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
    check_sample_credentials()

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
            'script': os.path.join('modules', 'image_server.py'),
            'title': 'kai - Image Server'
        },
        {
            'name': 'Control Server',
            'script': os.path.join('modules', 'control_server.py'),
            'title': 'kai - Control Server'
        },
        {
            'name': 'MQTT Monitor',
            'script': os.path.join('tools', 'monitor_mqtt.py'),
            'title': 'kai - MQTT Monitor'
        },
        {
            'name': 'HTTP API Server',
            'script': os.path.join('modules', 'http_api.py'),
            'title': 'kai - HTTP API'
        },
    ]
    
    # Launch modules
    print(colored("Launching modules...", Colors.CYAN))
    print()
    
    processes = launch_modules(modules)
    
    print()
    print(colored("=" * 70, Colors.CYAN))
    print(colored(f"[OK] Launched {len(processes)} backend modules successfully!", Colors.GREEN + Colors.BOLD))
    print(colored("=" * 70, Colors.CYAN))
    print()
    print(colored("System is starting up...", Colors.CYAN))
    print()
    print(colored("Modules launched:", Colors.BOLD))
    for module in modules:
        print(f"  • {module['name']}")
    print()
    print(colored("To launch the UI dashboard:", Colors.GREEN + Colors.BOLD))
    print("  Run 'cd dashboard; flutter run' in a new terminal to start the Flutter Dashboard UI!")
    print()
    print(colored("To stop the system:", Colors.YELLOW))
    print("  1. Close each terminal window, OR")
    print("  2. Press Ctrl+C in each window")
    print()
    print(colored("Tip: Check the MQTT Monitor window to see real-time message traffic!", Colors.CYAN))
    print()
    
    # Load config to check for DEV_MODE warning
    from modules.config_manager import get_config_namespace
    config = get_config_namespace()
    is_dev_mode = getattr(config, "DEV_MODE", False)
    
    # Keep this script running
    try:
        if is_dev_mode:
            print()
            print(colored("┌────────────────────────────────────────────────────────────┐", Colors.RED + Colors.BOLD))
            print(colored("│ WARNING: SYSTEM IS RUNNING IN AN UNSAFE / DEVELOPMENT MODE │", Colors.RED + Colors.BOLD))
            print(colored("│ - Using shared kai_admin / legacy developer credentials.   │", Colors.RED))
            print(colored("│ - MQTT feed privileges are NOT isolated.                   │", Colors.RED))
            print(colored("│ Please run configure_kai.py to secure this deployment!     │", Colors.RED))
            print(colored("└────────────────────────────────────────────────────────────┘", Colors.RED + Colors.BOLD))
            print()
            
        print(colored("Press Ctrl+C here to exit this launcher (modules will continue running)", Colors.YELLOW))
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print(colored("\n\nLauncher stopped. Modules are still running.", Colors.YELLOW))
        print(colored("Close individual terminal windows to stop each module.", Colors.YELLOW))


if __name__ == "__main__":
    main()
