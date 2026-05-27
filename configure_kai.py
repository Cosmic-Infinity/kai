import os
import sys
import subprocess
import getpass
import json
import ctypes
import platform
import time

# Add modules/ directory to import path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "modules"))
from config_manager import load_config, save_config

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

def configure_mosquitto_broker(username, password):
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
        
        print("[2/5] Creating password file...")
        passwd_path = "passwd"
        if os.path.exists(passwd_path):
            try:
                os.remove(passwd_path)
            except Exception as e:
                print(colored(f"[WARN] Failed to delete existing passwd file: {e}", Colors.YELLOW))
                
        if password:
            res = subprocess.run(["mosquitto_passwd.exe", "-c", "-b", "passwd", username, password], capture_output=True, text=True)
            if res.returncode == 0:
                print(colored("[OK] Password file created successfully.", Colors.GREEN))
                # Reset permissions so the LocalSystem account can read it when running as a service
                subprocess.run(["icacls", "passwd", "/reset"], capture_output=True)
            else:
                print(colored(f"[FAIL] Failed to create password file: {res.stderr}", Colors.RED))
                print("Attempting to restart service...")
                subprocess.run(["net", "start", "mosquitto"], capture_output=True)
                return False
        else:
            print(colored("[WARN] Skipping Mosquitto password file creation (No password provided).", Colors.YELLOW))
            
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
        
        if "KAI Dashboard System custom configuration" in content:
            # We already have our block, let's update it to ensure correct auth level
            print(colored("[OK] Custom KAI configuration listener already present. Updating auth settings...", Colors.GREEN))
            if password:
                content = content.replace("allow_anonymous true", "allow_anonymous false")
                if "password_file" not in content:
                    content += "\npassword_file C:\\Program Files\\mosquitto\\passwd\n"
            else:
                content = content.replace("allow_anonymous false", "allow_anonymous true")
        else:
            print("Appending custom listener configuration to mosquitto.conf...")
            custom_config = (
                "\n\n"
                "# =================================================================\n"
                "# KAI Dashboard System custom configuration\n"
                "# =================================================================\n"
                "listener 1883 0.0.0.0\n"
            )
            if password:
                custom_config += "allow_anonymous false\n"
                custom_config += "password_file C:\\Program Files\\mosquitto\\passwd\n"
            else:
                custom_config += "allow_anonymous true\n"
                
            content += custom_config
            print(colored("[OK] Custom listener configuration appended.", Colors.GREEN))
            
        with open(conf_path, "w", encoding="utf-8") as f:
            f.write(content)
            
        print("[4/5] Saving credentials to config.json SSoT...")
        # Handled in caller
        
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
    elevate_privileges()
    
    print(colored("\n==============================================", Colors.HEADER))
    # Keep KAI all caps
    print(colored("          Set Custom System Credentials       ", Colors.HEADER + Colors.BOLD))
    print(colored("==============================================", Colors.HEADER))
    
    username = input("\nEnter MQTT Username [kai_admin]: ").strip() or "kai_admin"
    password = getpass.getpass("Enter MQTT Password (Optional): ").strip()
    
    if not password:
        print(colored("\n[SECURITY WARNING] No MQTT Password provided!", Colors.YELLOW))
        print(colored("The Mosquitto broker will be configured with 'allow_anonymous true'.", Colors.YELLOW))
        print(colored("Anyone on your local network will be able to connect to the broker.", Colors.YELLOW))
    else:
        confirm_password = getpass.getpass("Confirm MQTT Password: ").strip()
        if password != confirm_password:
            print(colored("[FAIL] Passwords do not match.", Colors.RED))
            return
        
    api_key = getpass.getpass("\nEnter Image Server HTTP API Key (Optional): ").strip() or ""
    
    if not api_key:
        print(colored("\n[SECURITY WARNING] No HTTP API Key provided!", Colors.YELLOW))
        print(colored("The HTTP API will fail open, allowing unauthenticated network access.", Colors.YELLOW))
        print(colored("Anyone on the network can view camera frames and update system configs.", Colors.YELLOW))
    
    # Save to config.json SSoT
    config = load_config()
    config["MQTT_USERNAME"] = username
    config["MQTT_PASSWORD"] = password
    config["API_KEY"] = api_key
    save_config(config)
    
    # Configure broker
    success = configure_mosquitto_broker(username, password)
    if success:
        show_credentials_card(
            config.get("MQTT_BROKER_HOST", "127.0.0.1"),
            config.get("MQTT_BROKER_PORT", 1883),
            username,
            password,
            api_key if api_key else "[No Key / Public]"
        )
        print(colored("[SUCCESS] Credentials configured successfully!", Colors.GREEN + Colors.BOLD))

def enable_test_mode():
    """Instantly seed test credentials and configure Mosquitto automatically."""
    elevate_privileges()
    
    print(colored("\n==============================================", Colors.HEADER))
    print(colored("        Seeding Developer Test Mode Keys      ", Colors.HEADER + Colors.BOLD))
    print(colored("==============================================", Colors.HEADER))
    print("Pre-configuring standard local developer credentials...")
    
    username = "kai_admin"
    password = "kai"
    api_key = "testtest"
    
    # Save to config.json SSoT
    config = load_config()
    config["MQTT_USERNAME"] = username
    config["MQTT_PASSWORD"] = password
    config["API_KEY"] = api_key
    save_config(config)
    
    # Configure broker
    success = configure_mosquitto_broker(username, password)
    if success:
        show_credentials_card(
            config.get("MQTT_BROKER_HOST", "127.0.0.1"),
            config.get("MQTT_BROKER_PORT", 1883),
            username,
            password,
            api_key
        )
        print(colored("[SUCCESS] Developer test mode enabled!", Colors.GREEN + Colors.BOLD))
        print(colored("You can now launch the backend using: python start.py", Colors.CYAN))
    else:
        print(colored("[FAIL] Test mode setup failed. Verify Mosquitto services.", Colors.RED))

def reset_passwords():
    """Independently reset the MQTT password or the HTTP API Key."""
    elevate_privileges()
    
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
    main()
