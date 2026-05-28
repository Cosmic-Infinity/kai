import os
import sys
import subprocess
import getpass
import json
import ctypes
import platform
import time

# Add modules/ directory to import path for runtime and static IDE resolution
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "modules"))
from modules.config_manager import load_config, save_config

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

def safe_print(text, color_code=None):
    """Safely print text handling Windows console encoding limitations."""
    if color_code:
        text = colored(text, color_code)
    try:
        print(text)
    except UnicodeEncodeError:
        # Fallback: strip ANSI escape sequences and replace box characters
        import re
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        text_clean = ansi_escape.sub('', text)
        replacements = {
            "╔": "+", "═": "-", "╗": "+",
            "╠": "+", "╩": "+", "╝": "+",
            "╚": "+", "║": "|", "╬": "+",
            "▲": "^", "▼": "v", "■": "*",
            "•": "*", "┌": "+", "┐": "+",
            "└": "+", "┘": "+", "│": "|",
            "─": "-"
        }
        for char, repl in replacements.items():
            text_clean = text_clean.replace(char, repl)
        try:
            print(text_clean.encode('ascii', 'ignore').decode('ascii'))
        except:
            pass

def ensure_production_secrets(config):
    """Ensure all required production secrets and usernames exist in config."""
    import secrets
    def gen_pass():
        return secrets.token_hex(16)
        
    changed = False
    
    defaults = {
        "MQTT_USER_IMAGE_SERVER": "kai_image_server",
        "MQTT_USER_CONTROL_SERVER": "kai_control_server",
        "MQTT_USER_MONITOR": "kai_monitor",
        "MQTT_USER_POWER_FEED": "kai_edge_device"
    }
    
    for key, default_user in defaults.items():
        if not config.get(key):
            config[key] = default_user
            changed = True
            
    pass_fields = [
        "MQTT_PASSWORD_IMAGE_SERVER",
        "MQTT_PASSWORD_CONTROL_SERVER",
        "MQTT_PASSWORD_MONITOR",
        "MQTT_PASSWORD_POWER_FEED"
    ]
    
    for key in pass_fields:
        if not config.get(key):
            config[key] = gen_pass()
            changed = True
            
    return changed

def get_short_path(long_path):
    """Get the 8.3 short path version of a long path on Windows."""
    import ctypes
    
    if platform.system() != 'Windows':
        return long_path
        
    try:
        buf = ctypes.create_unicode_buffer(260)
        res = ctypes.windll.kernel32.GetShortPathNameW(long_path, buf, 260)
        if res > 0:
            return buf.value
    except:
        pass
    
    return long_path.replace("C:\\Program Files", "C:\\PROGRA~1")

def is_admin():
    """Check if the script is running with administrative privileges."""
    if platform.system() != 'Windows':
        return os.getuid() == 0
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

def elevate_privileges():
    """Relaunch the script with Administrator privileges on Windows."""
    if is_admin():
        return True
    
    if platform.system() == 'Windows':
        print(colored("Privilege Elevation Required to manage Mosquitto Windows service...", Colors.YELLOW))
        print("A User Account Control (UAC) prompt will appear shortly. Please approve it.")
        time.sleep(1.5)
        # Relaunch script as admin
        ctypes.windll.shell32.ShellExecuteW(
            None, "runas", sys.executable, " ".join(sys.argv), None, 1
        )
        sys.exit(0)
    else:
        print(colored("[FAIL] This script must be run as root. Please run using: sudo python configure_kai.py", Colors.RED))
        sys.exit(1)

def configure_mosquitto_broker(config):
    """Configure local Mosquitto MQTT broker with auth credentials."""
    print(colored("\n--- Configuring Mosquitto MQTT Broker ---", Colors.CYAN))
    
    if platform.system() != 'Windows':
        print(colored("[INFO] Non-Windows OS detected. Please configure your Mosquitto auth manually.", Colors.YELLOW))
        return True

    mosquitto_dir = r"C:\Program Files\mosquitto"
    if not os.path.exists(mosquitto_dir):
        print(colored(f"[WARN] Mosquitto directory not found at: {mosquitto_dir}", Colors.YELLOW))
        print(colored("Please ensure Mosquitto Broker is installed and update passwd file manually.", Colors.YELLOW))
        return False
        
    orig_dir = os.getcwd()
    try:
        print("[1/5] Stopping Mosquitto Broker service to release file locks...")
        os.chdir(mosquitto_dir)
        subprocess.run(["net", "stop", "mosquitto"], capture_output=True)
        
        is_dev_mode = config.get("DEV_MODE", False)
        if not is_dev_mode:
            if ensure_production_secrets(config):
                print(colored("[INFO] Populated missing secure production isolated server keys.", Colors.CYAN))
                save_config(config)
        
        print("[2/5] Creating password file...")
        passwd_path = "passwd"
        if os.path.exists(passwd_path):
            try:
                os.remove(passwd_path)
            except Exception as e:
                print(colored(f"[WARN] Failed to delete existing passwd file: {e}", Colors.YELLOW))
                
        if is_dev_mode:
            # Dev Mode - Single shared user (UNSAFE)
            username = config.get("MQTT_USERNAME", "kai_admin")
            password = config.get("MQTT_PASSWORD", "kai")
            
            res = subprocess.run(["mosquitto_passwd.exe", "-c", "-b", "passwd", username, password], capture_output=True, text=True)
            if res.returncode == 0:
                print(colored(f"[OK] Dev Admin user '{username}' created successfully.", Colors.GREEN))
                
                # Write the Access Control List (ACL) file
                acl_content = (
                    "# =================================================================\n"
                    "# kai System Access Control Lists (DEVELOPER UNSAFE MODE)\n"
                    "# =================================================================\n\n"
                    f"user {username}\n"
                    "topic readwrite kai/power\n"
                    "topic readwrite kai/control\n"
                    "topic readwrite kai/force_request\n"
                    "topic readwrite kai/force_served\n"
                )
                with open("acl", "w", encoding="utf-8") as f:
                    f.write(acl_content)
                print(colored("[WARN] Dev Mode ACL rules generated (UNSAFE).", Colors.YELLOW))
                subprocess.run(["icacls", "passwd", "/reset"], capture_output=True)
                subprocess.run(["icacls", "acl", "/reset"], capture_output=True)
            else:
                print(colored(f"[FAIL] Failed to create password file: {res.stderr}", Colors.RED))
                return False
        else:
            # Production Mode - Multiple Isolated Users
            dash_user = config.get("MQTT_USERNAME", "kai_dashboard")
            dash_pass = config.get("MQTT_PASSWORD")
            
            img_user = config.get("MQTT_USER_IMAGE_SERVER", "kai_image_server")
            img_pass = config.get("MQTT_PASSWORD_IMAGE_SERVER")
            
            ctrl_user = config.get("MQTT_USER_CONTROL_SERVER", "kai_control_server")
            ctrl_pass = config.get("MQTT_PASSWORD_CONTROL_SERVER")
            
            mon_user = config.get("MQTT_USER_MONITOR", "kai_monitor")
            mon_pass = config.get("MQTT_PASSWORD_MONITOR")
            
            pow_user = config.get("MQTT_USER_POWER_FEED", "kai_edge_device")
            pow_pass = config.get("MQTT_PASSWORD_POWER_FEED")
            
            # Create database and add dashboard user
            res = subprocess.run(["mosquitto_passwd.exe", "-c", "-b", "passwd", dash_user, dash_pass], capture_output=True, text=True)
            if res.returncode == 0:
                print(colored(f"[OK] Dashboard user '{dash_user}' created successfully.", Colors.GREEN))
                
                # Append internal users with strict return code verification
                res = subprocess.run(["mosquitto_passwd.exe", "-b", "passwd", img_user, img_pass], capture_output=True, text=True)
                if res.returncode != 0:
                    print(colored(f"[FAIL] Failed to create Image Server user: {res.stderr}", Colors.RED))
                    return False
                print(colored(f"[OK] Image Server user '{img_user}' created successfully.", Colors.GREEN))
                
                res = subprocess.run(["mosquitto_passwd.exe", "-b", "passwd", ctrl_user, ctrl_pass], capture_output=True, text=True)
                if res.returncode != 0:
                    print(colored(f"[FAIL] Failed to create Control Server user: {res.stderr}", Colors.RED))
                    return False
                print(colored(f"[OK] Control Server user '{ctrl_user}' created successfully.", Colors.GREEN))
                
                res = subprocess.run(["mosquitto_passwd.exe", "-b", "passwd", mon_user, mon_pass], capture_output=True, text=True)
                if res.returncode != 0:
                    print(colored(f"[FAIL] Failed to create Traffic Monitor user: {res.stderr}", Colors.RED))
                    return False
                print(colored(f"[OK] Traffic Monitor user '{mon_user}' created successfully.", Colors.GREEN))
                
                res = subprocess.run(["mosquitto_passwd.exe", "-b", "passwd", pow_user, pow_pass], capture_output=True, text=True)
                if res.returncode != 0:
                    print(colored(f"[FAIL] Failed to create Connected Devices (Power Feed) user: {res.stderr}", Colors.RED))
                    return False
                print(colored(f"[OK] Connected Devices (Power Feed) user '{pow_user}' created successfully.", Colors.GREEN))
                
                # Write the Access Control List (ACL) file
                acl_content = (
                    "# =================================================================\n"
                    "# kai System Access Control Lists (PRODUCTION ISOLATED MODE)\n"
                    "# =================================================================\n\n"
                    f"user {img_user}\n"
                    "topic read kai/force_request\n"
                    "topic write kai/force_served\n\n"
                    f"user {ctrl_user}\n"
                    "topic read kai/control\n"
                    "topic write kai/power\n\n"
                    f"user {pow_user}\n"
                    "topic read kai/power\n\n"
                    f"user {mon_user}\n"
                    "topic read kai/#\n\n"
                    f"user {dash_user}\n"
                    "topic read kai/power\n"
                    "topic read kai/force_served\n"
                    "topic write kai/control\n"
                    "topic write kai/force_request\n"
                )
                with open("acl", "w", encoding="utf-8") as f:
                    f.write(acl_content)
                print(colored("[OK] Strict production isolated ACL rules generated.", Colors.GREEN))
                subprocess.run(["icacls", "passwd", "/reset"], capture_output=True)
                subprocess.run(["icacls", "acl", "/reset"], capture_output=True)
            else:
                print(colored(f"[FAIL] Failed to create password file: {res.stderr}", Colors.RED))
                return False
            
        print("[3/5] Updating mosquitto.conf...")
        conf_path = "mosquitto.conf"
        if not os.path.exists(conf_path):
            print(colored("[FAIL] mosquitto.conf not found!", Colors.RED))
            return False
            
        with open(conf_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            
        custom_config_header = "# =================================================================\n# kai Dashboard System custom configuration"
        
        # Clean up any old custom configuration block entirely to prevent duplicate appends
        if custom_config_header in content:
            print(colored("[OK] Custom kai configuration listener found. Rewriting it cleanly...", Colors.GREEN))
            # Truncate content at the start of the kai custom block
            content = content.split(custom_config_header)[0].strip()
        else:
            print("Appending custom listener configuration to mosquitto.conf...")
            content = content.strip()
            
        # Clean any legacy kai configuration lines written outside the block, if any
        lines_clean = []
        for line in content.splitlines():
            if any(term in line for term in ["password_file", "acl_file"]) and "mosquitto" in line.lower() and not line.strip().startswith("#"):
                continue
            lines_clean.append(line)
        content = "\n".join(lines_clean).strip() + "\n\n"
            
        passwd_fwd_path = os.path.join(mosquitto_dir, "passwd").replace("\\", "/")
        acl_fwd_path = os.path.join(mosquitto_dir, "acl").replace("\\", "/")
        
        custom_config = (
            "# =================================================================\n"
            "# kai Dashboard System custom configuration\n"
            "# =================================================================\n"
            "listener 1883 0.0.0.0\n"
            "allow_anonymous false\n"
            f"password_file {passwd_fwd_path}\n"
            f"acl_file {acl_fwd_path}\n"
        )
        content += custom_config
        print(colored("[OK] Custom listener and ACL configuration successfully written.", Colors.GREEN))
            
        with open(conf_path, "w", encoding="utf-8") as f:
            f.write(content)
            
        print("[4/5] Saving credentials to config.json SSoT...")
        save_config(config)
        
        print("[5/5] Restarting Mosquitto Broker service...")
        res = subprocess.run(["net", "start", "mosquitto"], capture_output=True, text=True)
        if res.returncode != 0 and "has already been started" not in res.stderr and "has already been started" not in res.stdout:
            print(colored(f"[FAIL] Mosquitto Broker failed to start: {res.stderr or res.stdout}", Colors.RED))
            return False
            
        print(colored("[OK] Mosquitto Broker is running and secured.", Colors.GREEN))
        return True
        
    except Exception as e:
        print(colored(f"[FAIL] Error configuring Mosquitto: {e}", Colors.RED))
        return False
    finally:
        os.chdir(orig_dir)

def show_credentials_card(host, port, username, password, api_key, edge_user=None, edge_pass=None, color=Colors.CYAN, title="kai SYSTEM CREDENTIALS"):
    """Render a premium formatted terminal card displaying credentials."""
    card_width = 62
    safe_print("")
    safe_print("╔" + "═" * card_width + "╗", color)
    safe_print("║" + title.center(card_width, " ") + "║", color + Colors.BOLD)
    safe_print("╠" + "═" * card_width + "╣", color)
    safe_print(f"║  MQTT Host:            {host.ljust(38)} ║", color)
    safe_print(f"║  MQTT Port:            {str(port).ljust(38)} ║", color)
    safe_print(f"║  MQTT Username:        {username.ljust(38)} ║", color)
    safe_print(f"║  MQTT Password:        {password.ljust(38)} ║", color)
    if edge_user is not None:
        safe_print(f"║  Edge Device User:     {edge_user.ljust(38)} ║", color)
    if edge_pass is not None:
        safe_print(f"║  Edge Device Pass:     {edge_pass.ljust(38)} ║", color)
    safe_print(f"║  Image Server API Key: {api_key.ljust(38)} ║", color)
    safe_print("╠" + "═" * card_width + "╣", color)
    safe_print("║" + " Dashboard creds for Flutter App. Edge Device creds for relays. ".center(card_width, " ") + "║", Colors.YELLOW)
    safe_print("╚" + "═" * card_width + "╝", color)
    safe_print("")

def check_backend_running():
    import requests
    try:
        requests.get("http://127.0.0.1:8000", timeout=1)
        return True
    except:
        return False

def warn_hot_swap():
    if check_backend_running():
        print(colored("\n[WARN] The backend system appears to be currently running!", Colors.YELLOW + Colors.BOLD))
        print(colored("Updating credentials will restart the Mosquitto broker and disconnect active feeds.", Colors.YELLOW))
        print(colored("The system CANNOT hot-reload credentials. You MUST restart the backend (start.py) afterwards.", Colors.YELLOW))
        confirm = input("Proceed with credential update? (y/n) [n]: ").strip().lower()
        if confirm != 'y':
            print(colored("Operation cancelled.", Colors.RED))
            return False
    return True

def set_custom_credentials():
    """Prompt user for custom credentials and save/configure them."""
    print_banner("Configure kai System Setup")
    
    if not warn_hot_swap():
        return
        
    config = load_config()
    
    print(colored("\n[PRODUCTION MODE] Initiating strict isolated user setup.", Colors.GREEN))
    username = input("Enter Dashboard MQTT Username [kai_dashboard]: ").strip() or "kai_dashboard"
    password = getpass.getpass("Enter Dashboard MQTT Password: ").strip()
    while not password:
        print(colored("[FAIL] Dashboard password cannot be empty in production mode.", Colors.RED))
        password = getpass.getpass("Enter Dashboard MQTT Password: ").strip()
    confirm_password = getpass.getpass("Confirm Dashboard MQTT Password: ").strip()
    if password != confirm_password:
        print(colored("[FAIL] Passwords do not match.", Colors.RED))
        return
    
    api_key = getpass.getpass("\nEnter Image Server HTTP API Key (Optional): ").strip() or ""
    
    edge_user = input("\nEnter Edge Device MQTT Username [kai_edge_device]: ").strip() or "kai_edge_device"
    edge_pass = getpass.getpass("Enter Edge Device MQTT Password: ").strip()
    while not edge_pass:
        print(colored("[FAIL] Edge Device password cannot be empty in production mode.", Colors.RED))
        edge_pass = getpass.getpass("Enter Edge Device MQTT Password: ").strip()
    
    # Randomly generate passwords for internal users
    import secrets
    def gen_pass():
        return secrets.token_hex(16) # 32-character secure random hex
        
    config["MQTT_USERNAME"] = username
    config["MQTT_PASSWORD"] = password
    config["API_KEY"] = api_key
    config["DEV_MODE"] = False
    
    config["MQTT_USER_IMAGE_SERVER"] = "kai_image_server"
    config["MQTT_PASSWORD_IMAGE_SERVER"] = gen_pass()
    
    config["MQTT_USER_CONTROL_SERVER"] = "kai_control_server"
    config["MQTT_PASSWORD_CONTROL_SERVER"] = gen_pass()
    
    config["MQTT_USER_MONITOR"] = "kai_monitor"
    config["MQTT_PASSWORD_MONITOR"] = gen_pass()
    
    config["MQTT_USER_POWER_FEED"] = edge_user
    config["MQTT_PASSWORD_POWER_FEED"] = edge_pass

    # Save to config.json SSoT
    save_config(config)
    
    # Configure broker
    success = configure_mosquitto_broker(config)
    if success:
        # Credentials card
        show_credentials_card(
            config.get("MQTT_BROKER_HOST", "127.0.0.1"),
            config.get("MQTT_BROKER_PORT", 1883),
            username,
            password,
            api_key if api_key else "[Optional]",
            edge_user=edge_user,
            edge_pass=edge_pass,
            color=Colors.GREEN,
            title="kai PRODUCTION CREDENTIALS"
        )
        print(colored("[SUCCESS] Credentials configured successfully!", Colors.GREEN + Colors.BOLD))

def print_banner(title):
    """Print a standardized beautiful cyan terminal banner."""
    width = 62
    safe_print("")
    safe_print("=" * width, Colors.CYAN)
    safe_print(title.center(width, " "), Colors.CYAN + Colors.BOLD)
    safe_print("=" * width, Colors.CYAN)

def enable_test_mode():
    """Instantly seed developer test credentials and configure Mosquitto automatically."""
    print_banner("Developer Mode Configuration")
    
    if not warn_hot_swap():
        return
        
    config = load_config()
    
    use_custom = input("\nDo you want to specify a custom developer password? (y/n) [n]: ").strip().lower() == "y"
    
    if use_custom:
        password = getpass.getpass("Enter Developer MQTT Password [kai_developer]: ").strip() or "kai_developer"
        api_key = getpass.getpass("Enter Image Server HTTP API Key [testtest]: ").strip() or "testtest"
    else:
        print("\nPre-configuring standard local developer credentials...")
        password = "kai_developer"
        api_key = "testtest"
        
    config["MQTT_USERNAME"] = "kai_admin"
    config["MQTT_PASSWORD"] = password
    config["API_KEY"] = api_key
    config["DEV_MODE"] = True
    
    # Remove any leftover production users from config
    for k in ["MQTT_USER_IMAGE_SERVER", "MQTT_PASSWORD_IMAGE_SERVER", 
              "MQTT_USER_CONTROL_SERVER", "MQTT_PASSWORD_CONTROL_SERVER",
              "MQTT_USER_MONITOR", "MQTT_PASSWORD_MONITOR",
              "MQTT_USER_POWER_FEED", "MQTT_PASSWORD_POWER_FEED"]:
        config.pop(k, None)
        
    save_config(config)
    
    # Configure broker
    success = configure_mosquitto_broker(config)
    if success:
        print(colored("\n┌────────────────────────────────────────────────────────────┐", Colors.RED + Colors.BOLD))
        print(colored("│ WARNING: SYSTEM IS NOW CONFIGURED IN AN UNSAFE DEV MODE!   │", Colors.RED + Colors.BOLD))
        print(colored("│ - Feeds access control is disabled (no client isolation).  │", Colors.RED))
        print(colored("│ - Legacy credentials (kai_admin) are active for all roles. │", Colors.RED))
        print(colored("└────────────────────────────────────────────────────────────┘", Colors.RED + Colors.BOLD))
        
        show_credentials_card(
            config.get("MQTT_BROKER_HOST", "127.0.0.1"),
            config.get("MQTT_BROKER_PORT", 1883),
            "kai_admin",
            password,
            api_key,
            edge_user="kai_admin",
            edge_pass=password
        )
        print(colored("[SUCCESS] Developer test mode enabled!", Colors.GREEN + Colors.BOLD))
        print(colored("You can now launch the backend using: python start.py", Colors.CYAN))

def reset_passwords():
    """Independently reset the MQTT password or the HTTP API Key."""
    print_banner("Reset Keys & Passwords")
    
    if not warn_hot_swap():
        return
        
    print("\nSelect what you want to reset:")
    print("1) MQTT Broker Credentials")
    print("2) Edge Device MQTT Credentials")
    print("3) HTTP Image Server API Key")
    print("4) Back to main menu")
    
    choice = input("\nSelect an option: ").strip()
    if choice == '1':
        config = load_config()
        if config.get("DEV_MODE", False):
            print(colored("\n[INFO] Dev Mode is currently active.", Colors.YELLOW))
            confirm = input("Transition system to secure Production (Isolated) Mode? (y/n) [y]: ").strip().lower() != "n"
            if not confirm:
                print(colored("\nReset cancelled. Custom isolated credentials require secure Production Mode.", Colors.RED))
                return
            config["DEV_MODE"] = False
            print(colored("[OK] Transitioning system to secure Production Mode...", Colors.GREEN))
        
        username = input("\nEnter New Dashboard MQTT Username [kai_dashboard]: ").strip() or "kai_dashboard"
        password = getpass.getpass("Enter New Dashboard MQTT Password: ").strip()
        while not password:
            print(colored("[FAIL] Dashboard password cannot be empty.", Colors.RED))
            password = getpass.getpass("Enter New Dashboard MQTT Password: ").strip()
        confirm_password = getpass.getpass("Confirm New Dashboard MQTT Password: ").strip()
        if password != confirm_password:
            print(colored("[FAIL] Passwords do not match.", Colors.RED))
            return
            
        config["MQTT_USERNAME"] = username
        config["MQTT_PASSWORD"] = password
        save_config(config)
        
        success = configure_mosquitto_broker(config)
        if success:
            show_credentials_card(
                config.get("MQTT_BROKER_HOST", "127.0.0.1"),
                config.get("MQTT_BROKER_PORT", 1883),
                username,
                password,
                config.get("API_KEY", "[Optional]"),
                edge_user=config.get("MQTT_USER_POWER_FEED"),
                edge_pass=config.get("MQTT_PASSWORD_POWER_FEED"),
                color=Colors.GREEN,
                title="UPDATED MQTT BROKER CREDENTIALS"
            )
            print(colored("[SUCCESS] Dashboard MQTT credentials updated successfully!", Colors.GREEN + Colors.BOLD))
            
    elif choice == '2':
        config = load_config()
        if config.get("DEV_MODE", False):
            print(colored("\n[INFO] Dev Mode is currently active.", Colors.YELLOW))
            confirm = input("Transition system to secure Production (Isolated) Mode? (y/n) [y]: ").strip().lower() != "n"
            if not confirm:
                print(colored("\nReset cancelled. Custom isolated credentials require secure Production Mode.", Colors.RED))
                return
            config["DEV_MODE"] = False
            print(colored("[OK] Transitioning system to secure Production Mode...", Colors.GREEN))
        
        edge_user = input("\nEnter Edge Device MQTT Username [kai_edge_device]: ").strip() or "kai_edge_device"
        edge_pass = getpass.getpass("Enter Edge Device MQTT Password: ").strip()
        while not edge_pass:
            print(colored("[FAIL] Edge Device password cannot be empty.", Colors.RED))
            edge_pass = getpass.getpass("Enter Edge Device MQTT Password: ").strip()
        
        config["MQTT_USER_POWER_FEED"] = edge_user
        config["MQTT_PASSWORD_POWER_FEED"] = edge_pass
        save_config(config)
        
        success = configure_mosquitto_broker(config)
        if success:
            show_credentials_card(
                config.get("MQTT_BROKER_HOST", "127.0.0.1"),
                config.get("MQTT_BROKER_PORT", 1883),
                config.get("MQTT_USERNAME", "kai_dashboard"),
                config.get("MQTT_PASSWORD", ""),
                config.get("API_KEY", "[Optional]"),
                edge_user=edge_user,
                edge_pass=edge_pass,
                color=Colors.GREEN,
                title="UPDATED EDGE DEVICE CREDENTIALS"
            )
            print(colored("[SUCCESS] Edge Device MQTT credentials updated successfully!", Colors.GREEN + Colors.BOLD))
            
    elif choice == '3':
        api_key = getpass.getpass("\nEnter new Image Server HTTP API Key (Optional): ").strip()
        
        if not api_key:
            print(colored("\n[SECURITY WARNING] No HTTP API Key provided!", Colors.YELLOW))
            print(colored("The HTTP API will fail open, allowing unauthenticated network access.", Colors.YELLOW))
            print(colored("Anyone on the network can view camera frames and update system configs.", Colors.YELLOW))
            
        config = load_config()
        config["API_KEY"] = api_key
        save_config(config)
        print(colored(f"\n[SUCCESS] API Key updated successfully! (New Key: {api_key if api_key else '[None / Public]'})", Colors.GREEN))
    elif choice == '4':
        return
    else:
        print(colored("[FAIL] Invalid choice.", Colors.RED))

def run_system_diagnostics():
    """Read config.json and verify that the broker and all configured feeds are running as expected."""
    print_banner("kai SYSTEM DIAGNOSTICS")
    
    config = load_config()
    host = config.get("MQTT_BROKER_HOST", "127.0.0.1")
    port = config.get("MQTT_BROKER_PORT", 1883)
    is_dev_mode = config.get("DEV_MODE", False)
    
    print(colored(f"[INFO] Initializing diagnostics...", Colors.CYAN))
    print(colored(f"  • Broker:    {host}:{port}", Colors.CYAN))
    print(colored(f"  • Deployment: {'DEVELOPMENT (UNSAFE)' if is_dev_mode else 'PRODUCTION (ISOLATED)'}", Colors.YELLOW if is_dev_mode else Colors.GREEN))
    print()
    
    # Define feeds/roles to check
    feeds_to_test = []
    if is_dev_mode:
        feeds_to_test.append({
            "name": "Legacy Developer Admin Feed",
            "username": "kai_admin",
            "password": config.get("MQTT_PASSWORD", "kai")
        })
    else:
        feeds_to_test.extend([
            {
                "name": "Dashboard UI Feed",
                "username": config.get("MQTT_USERNAME", "kai_dashboard"),
                "password": config.get("MQTT_PASSWORD", "kai")
            },
            {
                "name": "Connected Edge Device Feed",
                "username": config.get("MQTT_USER_POWER_FEED", "kai_edge_device"),
                "password": config.get("MQTT_PASSWORD_POWER_FEED")
            },
            {
                "name": "Image Server Feed",
                "username": config.get("MQTT_USER_IMAGE_SERVER", "kai_image_server"),
                "password": config.get("MQTT_PASSWORD_IMAGE_SERVER")
            },
            {
                "name": "Control Server Feed",
                "username": config.get("MQTT_USER_CONTROL_SERVER", "kai_control_server"),
                "password": config.get("MQTT_PASSWORD_CONTROL_SERVER")
            },
            {
                "name": "Traffic Monitor Feed",
                "username": config.get("MQTT_USER_MONITOR", "kai_monitor"),
                "password": config.get("MQTT_PASSWORD_MONITOR")
            }
        ])
        
    # Helper to test a single MQTT feed connection
    import paho.mqtt.client as mqtt
    
    def test_feed_conn(feed):
        username = feed["username"]
        password = feed["password"]
        
        connected = [False]
        conn_rc = [None]
        
        def on_connect(client, userdata, flags, rc, properties=None):
            conn_rc[0] = rc
            connected[0] = (rc == 0)
            client.disconnect()
            
        client = mqtt.Client(
            client_id=f"diag_test_{username[:10]}",
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2
        )
        client.on_connect = on_connect
        if username or password:
            client.username_pw_set(username, password)
            
        try:
            client.connect(host, port, keepalive=5)
            client.loop_start()
            time.sleep(0.8)
            client.loop_stop()
        except Exception as e:
            conn_rc[0] = str(e)
            
        return connected[0], conn_rc[0]

    # Test MQTT feeds
    print(colored("--- Step 1: Testing MQTT Feeds Authentication ---", Colors.CYAN))
    mqtt_results = {}
    for feed in feeds_to_test:
        name = feed["name"]
        user = feed["username"]
        print(f"Testing connection for '{name}' (User: '{user}')...", end="", flush=True)
        success, code = test_feed_conn(feed)
        if success:
            print(colored(" [PASS]", Colors.GREEN))
            mqtt_results[name] = ("PASS", "Connected and Authenticated")
        else:
            reason = f"Connection failed (Code/Error: {code})"
            print(colored(f" [FAIL] - {reason}", Colors.RED))
            mqtt_results[name] = ("FAIL", reason)
            
    # Test HTTP API
    print(colored("\n--- Step 2: Testing HTTP API Integration ---", Colors.CYAN))
    api_results = {}
    api_url = "http://127.0.0.1:8000"
    api_key = config.get("API_KEY", "")
    
    print(f"Checking HTTP API Server status at {api_url}...", end="", flush=True)
    import requests
    import subprocess
    import sys
    
    api_proc = None
    
    # Try connecting. If it fails, start the server temporarily.
    try:
        requests.get(api_url, timeout=1)
    except:
        print(colored(" [OFFLINE]", Colors.YELLOW))
        print(colored("  Temporarily starting HTTP API server for diagnostics...", Colors.CYAN))
        api_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modules", "http_api.py")
        api_proc = subprocess.Popen([sys.executable, api_script], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(3)  # Give it a moment to boot
        print(f"Checking HTTP API Server status at {api_url} again...", end="", flush=True)

    try:
        # Check anonymous GET to /api/cameras
        r_anon = requests.get(f"{api_url}/api/cameras", timeout=2)
        if api_key:
            if r_anon.status_code == 401:
                api_results["Anonymous Block"] = ("PASS", "Anonymous access blocked correctly with 401")
            else:
                api_results["Anonymous Block"] = ("FAIL", f"Anonymous access returned {r_anon.status_code} instead of 401!")
        else:
            api_results["Anonymous Block"] = ("WARN", "No API_KEY defined; system is open to public access")
            
        # Check authenticated GET to /api/cameras
        headers = {"X-API-Key": api_key} if api_key else {}
        r_auth = requests.get(f"{api_url}/api/cameras", headers=headers, timeout=2)
        if r_auth.status_code == 200:
            api_results["Authenticated Access"] = ("PASS", "Successfully authenticated and retrieved camera feed")
        else:
            api_results["Authenticated Access"] = ("FAIL", f"Authenticated GET returned status code {r_auth.status_code}")
            
        # Check config masking (no credentials leaked)
        r_config = requests.get(f"{api_url}/api/config", headers=headers, timeout=2)
        if r_config.status_code == 200:
            config_data = r_config.json()
            sensitive_keys = {
                "MQTT_PASSWORD", "MQTT_PASSWORD_IMAGE_SERVER", "MQTT_PASSWORD_CONTROL_SERVER",
                "MQTT_PASSWORD_MONITOR", "API_KEY", "MQTT_BROKER_PORT", "MQTT_BROKER_HOST"
            }
            leaks = [k for k in sensitive_keys if k in config_data]
            if leaks:
                api_results["Config Exposure Masking"] = ("FAIL", f"Sensitive fields exposed: {', '.join(leaks)}")
            else:
                api_results["Config Exposure Masking"] = ("PASS", "Config response masked perfectly (no secrets leaked)")
        else:
            api_results["Config Exposure Masking"] = ("FAIL", f"Config GET returned status code {r_config.status_code}")
            
        print(colored(" [ONLINE]", Colors.GREEN))
        
    except Exception as e:
        print(colored(" [OFFLINE]", Colors.RED))
        api_results["API Server Status"] = ("FAIL", f"HTTP API Server is unreachable: {e}")
    finally:
        if api_proc:
            print(colored("  Shutting down temporary HTTP API server...", Colors.CYAN))
            api_proc.terminate()
            api_proc.wait()
        
    # Render Diagnostics Card
    card_width = 62
    safe_print("")
    safe_print("╔" + "═" * card_width + "╗", Colors.CYAN)
    safe_print("║" + " kai IoT SYSTEM INTEGRITY DIAGNOSTICS CARD ".center(card_width, " ") + "║", Colors.CYAN + Colors.BOLD)
    safe_print("╠" + "═" * card_width + "╣", Colors.CYAN)
    
    # Render MQTT results
    safe_print("║  MQTT BROKER FEEDS STATUS:".ljust(card_width + 1) + "║", Colors.CYAN)
    for name, (status, desc) in mqtt_results.items():
        status_color = Colors.GREEN if status == "PASS" else Colors.RED
        status_label = f"[{status}]"
        line_txt = f"  • {name}:"
        spaces = card_width - len(line_txt) - len(status_label) - 2
        safe_print(colored(f"║{line_txt}{' ' * spaces}", Colors.CYAN) + colored(status_label, status_color + Colors.BOLD) + colored(" ║", Colors.CYAN))
        
    # Render HTTP API results
    safe_print("║".ljust(card_width + 1) + "║", Colors.CYAN)
    safe_print("║  HTTP API GATEWAY STATUS:".ljust(card_width + 1) + "║", Colors.CYAN)
    if "API Server Status" in api_results or not api_results:
        status_label = "[FAIL]"
        line_txt = "  • HTTP API Gateway Service:"
        spaces = card_width - len(line_txt) - len(status_label) - 2
        safe_print(colored(f"║{line_txt}{' ' * spaces}", Colors.CYAN) + colored(status_label, Colors.RED + Colors.BOLD) + colored(" ║", Colors.CYAN))
        desc = api_results.get("API Server Status", ("FAIL", "Server is Offline"))[1]
        desc_wrapped = desc[:50]
        desc_line = f"    Reason: {desc_wrapped}"
        safe_print(colored(f"║{desc_line.ljust(card_width)}║", Colors.RED))
    else:
        for name, (status, desc) in api_results.items():
            if status == "PASS":
                status_color = Colors.GREEN
                status_label = "[PASS]"
            elif status == "WARN":
                status_color = Colors.YELLOW
                status_label = "[WARN]"
            else:
                status_color = Colors.RED
                status_label = "[FAIL]"
            line_txt = f"  • {name}:"
            spaces = card_width - len(line_txt) - len(status_label) - 2
            safe_print(colored(f"║{line_txt}{' ' * spaces}", Colors.CYAN) + colored(status_label, status_color + Colors.BOLD) + colored(" ║", Colors.CYAN))
            
    safe_print("╠" + "═" * card_width + "╣", Colors.CYAN)
    
    # Overall summary
    all_pass = all(status == "PASS" for status, _ in mqtt_results.values()) and all(status in ("PASS", "WARN") for status, _ in api_results.values())
    if all_pass:
        summary_text = "ALL SYSTEM FEEDS & API CHANNELS OPERATING SUCCESSFULLY!"
        summary_color = Colors.GREEN
    else:
        summary_text = "INTEGRITY FAULTS DETECTED! VERIFY BROKER AND DEPLOYMENT RULES."
        summary_color = Colors.RED
        
    safe_print("║" + summary_text.center(card_width, " ") + "║", summary_color + Colors.BOLD)
    safe_print("╚" + "═" * card_width + "╝", Colors.CYAN)
    safe_print("")
    
    input("Press Enter to return to main menu...")

def main():
    while True:
        # Clear console color
        print(Colors.ENDC, end="")
        
        print_banner("kai IoT System Wizard")
        print("1) Set Custom Credentials (First-Time Setup)")
        print("2) Reset Passwords & Server Keys")
        print("3) Enable Developer Test Mode (Pre-seed keys)")
        print("4) Run System Diagnostics (Verify Active Feeds)")
        print("5) Exit")
        
        choice = input("\nSelect an option (1-5): ").strip()
        if choice == '1':
            set_custom_credentials()
        elif choice == '2':
            reset_passwords()
        elif choice == '3':
            enable_test_mode()
            # Pause to let the user see the keys before continuing
            input("\nPress Enter to return to main menu...")
        elif choice == '4':
            run_system_diagnostics()
        elif choice == '5':
            print(colored("\nExiting kai Wizard. Goodbye!", Colors.GREEN))
            break
        else:
            print(colored("\n[FAIL] Invalid choice. Please select from 1-5.", Colors.RED))

if __name__ == "__main__":
    # Ensure terminal has ANSI escape sequences active
    if platform.system() == 'Windows':
        os.system('')
    
    # Prompt before UAC elevation to ensure non-intrusive flow
    if not is_admin():
        print_banner("kai SYSTEM CONFIGURATION WIZARD")
        print(colored("\n[INFO] Setting up the kai System requires administrative privileges.", Colors.YELLOW))
        print("This is necessary to install/start the Mosquitto MQTT broker Windows service,")
        print("create system password registries, and configure restricted access permissions.")
        print()
        response = input(colored("Do you want to elevate privileges and start the configuration wizard? (y/n) [y]: ", Colors.BOLD)).strip().lower()
        if response == 'n':
            print(colored("\nSetup cancelled. Administrative privileges are required to configure kai.", Colors.RED))
            sys.exit(0)
            
    # Elevate privileges and launch menu
    elevate_privileges()
    main()
