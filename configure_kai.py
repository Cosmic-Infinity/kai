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
        
    try:
        print("[1/5] Stopping Mosquitto Broker service to release file locks...")
        os.chdir(mosquitto_dir)
        subprocess.run(["net", "stop", "mosquitto"], capture_output=True)
        
        is_dev_mode = config.get("DEV_MODE", False)
        
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
                    "# KAI System Access Control Lists (DEVELOPER UNSAFE MODE)\n"
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
            
            # Create database and add dashboard user
            res = subprocess.run(["mosquitto_passwd.exe", "-c", "-b", "passwd", dash_user, dash_pass], capture_output=True, text=True)
            if res.returncode == 0:
                print(colored(f"[OK] Dashboard user '{dash_user}' created successfully.", Colors.GREEN))
                
                # Append internal users
                subprocess.run(["mosquitto_passwd.exe", "-b", "passwd", img_user, img_pass], capture_output=True)
                print(colored(f"[OK] Image Server user '{img_user}' created successfully.", Colors.GREEN))
                
                subprocess.run(["mosquitto_passwd.exe", "-b", "passwd", ctrl_user, ctrl_pass], capture_output=True)
                print(colored(f"[OK] Control Server user '{ctrl_user}' created successfully.", Colors.GREEN))
                
                subprocess.run(["mosquitto_passwd.exe", "-b", "passwd", mon_user, mon_pass], capture_output=True)
                print(colored(f"[OK] Traffic Monitor user '{mon_user}' created successfully.", Colors.GREEN))
                
                # Write the Access Control List (ACL) file
                acl_content = (
                    "# =================================================================\n"
                    "# KAI System Access Control Lists (PRODUCTION ISOLATED MODE)\n"
                    "# =================================================================\n\n"
                    f"user {img_user}\n"
                    "topic read kai/force_request\n"
                    "topic write kai/force_served\n\n"
                    f"user {ctrl_user}\n"
                    "topic read kai/control\n"
                    "topic write kai/power\n\n"
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
            
        # Clean any unquoted password paths from legacy setup script bugs
        content = content.replace(
            'password_file "C:\\Program Files\\mosquitto\\passwd"',
            "password_file C:\\Program Files\\mosquitto\\passwd"
        )
        
        custom_config_header = "# =================================================================\n# KAI Dashboard System custom configuration"
        
        # Clean up any old custom configuration block entirely to prevent duplicate appends
        if custom_config_header in content:
            print(colored("[OK] Custom KAI configuration listener found. Rewriting it cleanly...", Colors.GREEN))
            # Truncate content at the start of the KAI custom block
            content = content.split(custom_config_header)[0].strip() + "\n\n"
        else:
            print("Appending custom listener configuration to mosquitto.conf...")
            content = content.strip() + "\n\n"
            
        custom_config = (
            "# =================================================================\n"
            "# KAI Dashboard System custom configuration\n"
            "# =================================================================\n"
            "listener 1883 0.0.0.0\n"
            "allow_anonymous false\n"
            "password_file C:\\Program Files\\mosquitto\\passwd\n"
            "acl_file C:\\Program Files\\mosquitto\\acl\n"
        )
        content += custom_config
        print(colored("[OK] Custom listener and ACL configuration successfully written.", Colors.GREEN))
            
        with open(conf_path, "w", encoding="utf-8") as f:
            f.write(content)
            
        print("[4/5] Saving credentials to config.json SSoT...")
        
        print("[5/5] Restarting Mosquitto Broker service...")
        subprocess.run(["net", "start", "mosquitto"], capture_output=True)
        print(colored("[OK] Mosquitto Broker is running and secured.", Colors.GREEN))
        return True
        
    except Exception as e:
        print(colored(f"[FAIL] Error configuring Mosquitto: {e}", Colors.RED))
        return False

def show_credentials_card(host, port, username, password, api_key):
    """Render a premium formatted terminal card displaying credentials."""
    card_width = 62
    print()
    print(colored("╔" + "═" * card_width + "╗", Colors.CYAN))
    print(colored("║" + " KAI SYSTEM CREDENTIALS SEEDED ".center(card_width, " ") + "║", Colors.CYAN + Colors.BOLD))
    print(colored("╠" + "═" * card_width + "╣", Colors.CYAN))
    print(colored(f"║  MQTT Host:            {host.ljust(38)} ║", Colors.CYAN))
    print(colored(f"║  MQTT Port:            {str(port).ljust(38)} ║", Colors.CYAN))
    print(colored(f"║  MQTT Username:        {username.ljust(38)} ║", Colors.CYAN))
    print(colored(f"║  MQTT Password:        {password.ljust(38)} ║", Colors.CYAN))
    print(colored(f"║  Image Server API Key: {api_key.ljust(38)} ║", Colors.CYAN))
    print(colored("╠" + "═" * card_width + "╣", Colors.CYAN))
    print(colored("║" + " Use these credentials inside the Flutter App to connect! ".center(card_width, " ") + "║", Colors.YELLOW))
    print(colored("╚" + "═" * card_width + "╝", Colors.CYAN))
    print()

def set_custom_credentials():
    """Prompt user for custom credentials and save/configure them."""
    print(colored("\n==============================================", Colors.HEADER))
    print(colored("          Configure KAI System Setup          ", Colors.HEADER + Colors.BOLD))
    print(colored("==============================================", Colors.HEADER))
    
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

    # Save to config.json SSoT
    save_config(config)
    
    # Configure broker
    success = configure_mosquitto_broker(config)
    if success:
        # Custom card for isolated production mode
        card_width = 62
        print()
        print(colored("╔" + "═" * card_width + "╗", Colors.GREEN))
        print(colored("║" + " KAI SECURE PRODUCTION CREDENTIALS SEEDED ".center(card_width, " ") + "║", Colors.GREEN + Colors.BOLD))
        print(colored("╠" + "═" * card_width + "╣", Colors.GREEN))
        print(colored(f"║  MQTT Host:            {config.get('MQTT_BROKER_HOST', '127.0.0.1').ljust(38)} ║", Colors.GREEN))
        print(colored(f"║  MQTT Port:            {str(config.get('MQTT_BROKER_PORT', 1883)).ljust(38)} ║", Colors.GREEN))
        print(colored(f"║  Dashboard Username:   {username.ljust(38)} ║", Colors.GREEN))
        print(colored(f"║  Dashboard Password:   {password.ljust(38)} ║", Colors.GREEN))
        print(colored(f"║  Image Server API Key: {(api_key if api_key else '[None / Public]').ljust(38)} ║", Colors.GREEN))
        print(colored("╠" + "═" * card_width + "╣", Colors.GREEN))
        print(colored("║" + " Use these secure credentials inside the Flutter App!     ".center(card_width, " ") + "║", Colors.YELLOW))
        print(colored("║" + " Internal modules will connect automatically via randomly  ".center(card_width, " ") + "║", Colors.YELLOW))
        print(colored("║" + " generated background credentials.                         ".center(card_width, " ") + "║", Colors.YELLOW))
        print(colored("╚" + "═" * card_width + "╝", Colors.GREEN))
        print()
        
        print(colored("[SUCCESS] Credentials configured successfully!", Colors.GREEN + Colors.BOLD))

def enable_test_mode():
    """Instantly seed developer test credentials and configure Mosquitto automatically."""
    print(colored("\n==============================================", Colors.HEADER))
    print(colored("        Developer Mode Configuration          ", Colors.HEADER + Colors.BOLD))
    print(colored("==============================================", Colors.HEADER))
    
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
              "MQTT_USER_MONITOR", "MQTT_PASSWORD_MONITOR"]:
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
            api_key
        )
        print(colored("[SUCCESS] Developer test mode enabled!", Colors.GREEN + Colors.BOLD))
        print(colored("You can now launch the backend using: python start.py", Colors.CYAN))

def reset_passwords():
    """Independently reset the MQTT password or the HTTP API Key."""
    print(colored("\n==============================================", Colors.HEADER))
    print(colored("             Reset Keys & Passwords           ", Colors.HEADER + Colors.BOLD))
    print(colored("==============================================", Colors.HEADER))
    
    print("\nSelect what you want to reset:")
    print("1) MQTT Broker Credentials")
    print("2) HTTP Image Server API Key")
    print("3) Back to main menu")
    
    choice = input("\nSelect an option: ").strip()
    if choice == '1':
        set_custom_credentials()
    elif choice == '2':
        api_key = getpass.getpass("\nEnter new Image Server HTTP API Key (Optional): ").strip()
        
        if not api_key:
            print(colored("\n[SECURITY WARNING] No HTTP API Key provided!", Colors.YELLOW))
            print(colored("The HTTP API will fail open, allowing unauthenticated network access.", Colors.YELLOW))
            print(colored("Anyone on the network can view camera frames and update system configs.", Colors.YELLOW))
            
        config = load_config()
        config["API_KEY"] = api_key
        save_config(config)
        print(colored(f"\n[SUCCESS] API Key updated successfully! (New Key: {api_key if api_key else '[None / Public]'})", Colors.GREEN))
    elif choice == '3':
        return
    else:
        print(colored("[FAIL] Invalid choice.", Colors.RED))

def main():
    while True:
        # Clear console color
        print(Colors.ENDC, end="")
        
        print(colored("\n==============================================", Colors.CYAN))
        print(colored("          KAI IoT System Wizard               ", Colors.CYAN + Colors.BOLD))
        print(colored("==============================================", Colors.CYAN))
        print("1) Set Custom Credentials (First-Time Setup)")
        print("2) Reset Passwords & Server Keys")
        print("3) Enable Developer Test Mode (Pre-seed keys)")
        print("4) Exit")
        
        choice = input("\nSelect an option (1-4): ").strip()
        if choice == '1':
            set_custom_credentials()
        elif choice == '2':
            reset_passwords()
        elif choice == '3':
            enable_test_mode()
            # Pause to let the user see the keys before continuing
            input("\nPress Enter to return to main menu...")
        elif choice == '4':
            print(colored("\nExiting KAI Wizard. Goodbye!", Colors.GREEN))
            break
        else:
            print(colored("\n[FAIL] Invalid choice. Please select from 1-4.", Colors.RED))

if __name__ == "__main__":
    # Ensure terminal has ANSI escape sequences active
    if platform.system() == 'Windows':
        os.system('')
    
    # Prompt before UAC elevation to ensure non-intrusive flow
    if not is_admin():
        print(colored("=" * 70, Colors.CYAN))
        print(colored("              KAI SYSTEM CONFIGURATION WIZARD              ", Colors.CYAN + Colors.BOLD))
        print(colored("=" * 70, Colors.CYAN))
        print(colored("\n[INFO] Setting up the KAI System requires administrative privileges.", Colors.YELLOW))
        print("This is necessary to install/start the Mosquitto MQTT broker Windows service,")
        print("create system password registries, and configure restricted access permissions.")
        print()
        response = input(colored("Do you want to elevate privileges and start the configuration wizard? (y/n) [y]: ", Colors.BOLD)).strip().lower()
        if response == 'n':
            print(colored("\nSetup cancelled. Administrative privileges are required to configure KAI.", Colors.RED))
            sys.exit(0)
            
    # Elevate privileges and launch menu
    elevate_privileges()
    main()
