import os
import sys
import subprocess
import getpass
import json

def get_workspace_dir():
    # Assuming this script is in tools/, the workspace is one level up
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def update_python_config(username, password):
    config_path = os.path.join(get_workspace_dir(), "mqtt_config.py")
    if not os.path.exists(config_path):
        print(f"[WARN] mqtt_config.py not found at {config_path}")
        return

    with open(config_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    with open(config_path, "w", encoding="utf-8") as f:
        for line in lines:
            if line.startswith("MQTT_USERNAME"):
                f.write(f'MQTT_USERNAME = "{username}"\n')
            elif line.startswith("MQTT_PASSWORD"):
                f.write(f'MQTT_PASSWORD = "{password}"\n')
            else:
                f.write(line)
    print("[OK] Updated mqtt_config.py")

def update_json_config(username, password):
    config_path = os.path.join(get_workspace_dir(), "client_config.json")
    if not os.path.exists(config_path):
        print(f"[WARN] client_config.json not found at {config_path}")
        return

    with open(config_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print("[WARN] Could not parse client_config.json")
            return

    data["MQTT_USERNAME"] = username
    data["MQTT_PASSWORD"] = password

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    print("[OK] Updated client_config.json")

def main():
    print("=====================================================")
    print("      Mosquitto MQTT Authentication Setup Utility    ")
    print("=====================================================\n")
    
    print("This tool requires Administrator privileges.")
    mosquitto_dir = r"C:\Program Files\mosquitto"
    if not os.path.exists(mosquitto_dir):
        print(f"[FAIL] Mosquitto directory not found at: {mosquitto_dir}")
        sys.exit(1)
        
    print("Please enter the credentials you want to use for the MQTT broker.\n")
    username = input("Username [kai_admin]: ").strip()
    if not username:
        username = "kai_admin"
        
    password = getpass.getpass("Password: ").strip()
    if not password:
        print("[FAIL] Password cannot be empty.")
        sys.exit(1)
        
    confirm_password = getpass.getpass("Confirm Password: ").strip()
    if password != confirm_password:
        print("[FAIL] Passwords do not match.")
        sys.exit(1)

    print("\n[1/5] Stopping Mosquitto Broker service to release file locks...")
    # Change to mosquitto dir for mosquitto_passwd
    os.chdir(mosquitto_dir)
    subprocess.run(["net", "stop", "mosquitto"], capture_output=True)
    
    print("[2/5] Securing password file...")
    passwd_path = "passwd"
    if os.path.exists(passwd_path):
        try:
            os.remove(passwd_path)
        except Exception as e:
            print(f"[WARN] Failed to delete existing passwd file: {e}")
            
    res = subprocess.run(["mosquitto_passwd.exe", "-c", "-b", "passwd", username, password], capture_output=True, text=True)
    if res.returncode == 0:
        print("[OK] Password file created successfully.")
        # Fix permissions so the LocalSystem account can read it when running as a service
        subprocess.run(["icacls", "passwd", "/reset"], capture_output=True)
    else:
        print(f"[FAIL] Failed to create password file: {res.stderr}")
        print("Re-starting service...")
        subprocess.run(["net", "start", "mosquitto"], capture_output=True)
        sys.exit(1)
        
    print("[3/5] Updating mosquitto.conf...")
    conf_path = "mosquitto.conf"
    if not os.path.exists(conf_path):
        print(f"[FAIL] mosquitto.conf not found!")
        sys.exit(1)
        
    with open(conf_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
        
    # Heal existing unquoted configuration if present (from previous bugs)
    content = content.replace(
        'password_file "C:\\Program Files\\mosquitto\\passwd"',
        "password_file C:\\Program Files\\mosquitto\\passwd"
    )
    
    if "listener 1883 0.0.0.0" in content or "listener 1883" in content:
        print("[OK] KAI configuration already present in mosquitto.conf.")
    else:
        print("Appending custom listener configuration to the end of mosquitto.conf...")
        custom_config = (
            "\n\n"
            "# =================================================================\n"
            "# KAI Dashboard System custom configuration\n"
            "# =================================================================\n"
            "listener 1883 0.0.0.0\n"
            "allow_anonymous false\n"
            "password_file C:\\Program Files\\mosquitto\\passwd\n"
        )
        content += custom_config
        print("[OK] Custom configuration appended.")
        
    with open(conf_path, "w", encoding="utf-8") as f:
        f.write(content)
        
    print(f"[4/5] Updating application configuration files...")
    update_python_config(username, password)
    update_json_config(username, password)
        
    print("[5/5] Starting Mosquitto Broker service...")
    subprocess.run(["net", "start", "mosquitto"], capture_output=True)
    
    print("\n=====================================================")
    print("SUCCESS! Mosquitto MQTT broker is now secured.")
    print(f"Username: {username}")
    print("Application configurations have been updated automatically.")
    print("=====================================================\n")

if __name__ == "__main__":
    main()
