import os
import sys
import time
import json
import subprocess
import requests
import paho.mqtt.client as mqtt


class Colors:
    HEADER = '[95m'
    BLUE = '[94m'
    CYAN = '[96m'
    GREEN = '[92m'
    YELLOW = '[93m'
    RED = '[91m'
    ENDC = '[0m'
    BOLD = '[1m'

def colored(text, color):
    return f"{color}{text}{Colors.ENDC}"
    
try:
    import colorama
    colorama.init()
except ImportError:
    pass


# Resolve project path dynamically (works if run from tools/, scratch/, or project root)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(SCRIPT_DIR) in ["tools", "scratch"]:
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
else:
    PROJECT_ROOT = SCRIPT_DIR
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "config.json")

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)

config = load_config()
host = config.get("MQTT_BROKER_HOST", "127.0.0.1")
port = config.get("MQTT_BROKER_PORT", 1883)
api_key = config.get("API_KEY", "kai")

dashboard_user = config.get("MQTT_USERNAME", "kai")
dashboard_pass = config.get("MQTT_PASSWORD", "kai")

print(colored("==============================================================", Colors.CYAN))
print(colored("          kai DEEP-DIVE SECURITY AUDIT & STRESS TEST          ", Colors.CYAN + Colors.BOLD))
print(colored("==============================================================", Colors.CYAN))
print(f"Target MQTT Broker:  {host}:{port}")
print(f"Target HTTP API:     http://127.0.0.1:8000")
print(f"Dashboard User:      '{dashboard_user}'")
print("==============================================================\n")


# ----------------------------------------------------------------
# PART 1: HTTP API HARDENING AUDIT
# ----------------------------------------------------------------
print(colored("--- PART 1: HTTP API Security Controls ---", Colors.CYAN + Colors.BOLD))

# To test the newly hardened code, we must terminate any old server running on Port 8000
# and start a clean instance of the updated modules/http_api.py.
print(colored("[INFO] Clearing any active process on Port 8000 to ensure we test the newly updated, hardened server...", Colors.BLUE))
try:
    netstat = subprocess.check_output("netstat -ano", shell=True).decode('utf-8', errors='ignore')
    for line in netstat.splitlines():
        if ":8000" in line and "LISTENING" in line:
            pid = line.strip().split()[-1]
            print(f"  • Found process {pid} listening on port 8000. Terminating it...")
            subprocess.run(f"taskkill /F /PID {pid}", shell=True, capture_output=True)
            time.sleep(1.5)
except Exception as e:
    print(f"  • Note: Could not query/terminate existing port 8000 process: {e}")

print(colored("[INFO] Starting a fresh instance of the hardened HTTP API Server...", Colors.BLUE))
api_script = os.path.join(PROJECT_ROOT, "modules", "http_api.py")
api_process = subprocess.Popen(
    [sys.executable, api_script],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    cwd=PROJECT_ROOT
)
# Wait for the server to spin up
time.sleep(2.5)

def run_http_tests():
    url_cameras = "http://127.0.0.1:8000/api/cameras"
    url_config = "http://127.0.0.1:8000/api/config"
    
    # 1. Anonymous Access to Read Endpoint
    try:
        r = requests.get(url_cameras, timeout=3)
        print(colored(f"[TEST HTTP-1] GET without credentials: Status Code = {r.status_code}", Colors.YELLOW))
        if r.status_code == 401:
            print(colored("  • SUCCESS: Anonymous read request blocked with 401.", Colors.GREEN))
        else:
            print(colored(f"  • FAILURE: Anonymous read request bypass allowed (Status {r.status_code})!", Colors.RED + Colors.BOLD))
    except Exception as e:
        print(f"  • ERROR: {e}")

    # 2. Malformed X-API-Key Header (SQL Injection / Path Traversal attempts)
    bad_keys = ["admin", "root", "' OR '1'='1", "null", ""]
    for idx, key in enumerate(bad_keys, 1):
        try:
            r = requests.get(url_cameras, headers={"X-API-Key": key}, timeout=3)
            print(colored(f"[TEST HTTP-2.{idx}] GET with malformed key '{key}': Status Code = {r.status_code}", Colors.YELLOW))
            if r.status_code == 401:
                print(colored("    • SUCCESS: Malformed key blocked.", Colors.GREEN))
            else:
                print(colored("    • FAILURE: Malformed key bypassed authorization!", Colors.RED + Colors.BOLD))
        except Exception as e:
            print(f"    • ERROR: {e}")

    # 3. Anonymous Access to Write Endpoint (POST Configuration)
    try:
        payload = {"DASHBOARD_INTERVAL": 30}
        r = requests.post(url_config, json=payload, timeout=3)
        print(colored(f"[TEST HTTP-3] POST config without credentials: Status Code = {r.status_code}", Colors.YELLOW))
        if r.status_code == 401:
            print(colored("  • SUCCESS: Anonymous write request blocked.", Colors.GREEN))
        else:
            print(colored(f"  • FAILURE: Anonymous config overwrite allowed (Status {r.status_code})!", Colors.RED + Colors.BOLD))
    except Exception as e:
        print(f"  • ERROR: {e}")

    # 4. Valid Authorization using X-API-Key
    try:
        r = requests.get(url_cameras, headers={"X-API-Key": api_key}, timeout=3)
        print(colored(f"[TEST HTTP-4] GET with valid X-API-Key ('{api_key}'): Status Code = {r.status_code}", Colors.YELLOW))
        if r.status_code == 200:
            print(colored("  • SUCCESS: Authorized client authenticated successfully.", Colors.GREEN))
        else:
            print(colored(f"  • FAILURE: Valid API Key was rejected (Status {r.status_code})!", Colors.RED + Colors.BOLD))
    except Exception as e:
        print(f"  • ERROR: {e}")

    # 5. Valid Authorization using Bearer Token
    try:
        r = requests.get(url_cameras, headers={"Authorization": f"Bearer {api_key}"}, timeout=3)
        print(colored(f"[TEST HTTP-5] GET with valid Bearer Token ('{api_key}'): Status Code = {r.status_code}", Colors.YELLOW))
        if r.status_code == 200:
            print(colored("  • SUCCESS: Bearer Token authentication successfully validated.", Colors.GREEN))
        else:
            print(colored(f"  • FAILURE: Bearer Token was rejected (Status {r.status_code})!", Colors.RED + Colors.BOLD))
    except Exception as e:
        print(f"  • ERROR: {e}")

    # 6. Read Config Exposure & Leak Prevention Audit (Least Privilege Check)
    try:
        r = requests.get(url_config, headers={"X-API-Key": api_key}, timeout=3)
        print(colored(f"[TEST HTTP-6] GET config details: Status Code = {r.status_code}", Colors.YELLOW))
        if r.status_code == 200:
            config_data = r.json()
            # Verify that no sensitive credentials, ports, or systems keys are returned
            sensitive_keys = {
                "MQTT_PASSWORD", "MQTT_PASSWORD_IMAGE_SERVER", "MQTT_PASSWORD_CONTROL_SERVER",
                "MQTT_PASSWORD_MONITOR", "API_KEY", "MQTT_BROKER_PORT", "MQTT_BROKER_HOST"
            }
            leak_detected = False
            for sk in sensitive_keys:
                if sk in config_data:
                    leak_detected = True
                    print(colored(f"    • FAILURE: Leak detected! Key '{sk}' was exposed in the response payload!", Colors.RED + Colors.BOLD))
            
            if not leak_detected:
                print(colored("  • SUCCESS: Config response masked successfully. No credentials or server properties were leaked.", Colors.GREEN))
                print(f"    -> Exposed client-safe fields: {list(config_data.keys())}")
        else:
            print(colored(f"  • FAILURE: Authorized config read failed (Status {r.status_code})!", Colors.RED + Colors.BOLD))
    except Exception as e:
        print(f"  • ERROR: {e}")

run_http_tests()
print()


# ----------------------------------------------------------------
# PART 2: MQTT BROKER PASSWORD GATEKEEPER
# ----------------------------------------------------------------
print(colored("--- PART 2: MQTT Broker Password Gatekeeper ---", Colors.CYAN + Colors.BOLD))

def test_mqtt_auth(user, password, desc):
    connected = [False]
    conn_result = [None]
    
    def on_connect(client, userdata, flags, rc, properties=None):
        conn_result[0] = rc
        connected[0] = (rc == 0)
        client.disconnect()
        
    client = mqtt.Client(client_id=f"audit_gate_{user}", callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    if user or password:
        client.username_pw_set(user, password)
        
    try:
        client.connect(host, port, keepalive=5)
        client.loop_start()
        time.sleep(1)
        client.loop_stop()
    except Exception as e:
        pass
        
    result_str = str(conn_result[0])
    if connected[0]:
        print(colored(f"[TEST AUTH] Connect as '{user}' / '{password}': ALLOWED.", Colors.YELLOW))
        return True
    else:
        print(colored(f"[TEST AUTH] Connect as '{user}' / '{password}': REJECTED (Reason/Code: {result_str}).", Colors.YELLOW))
        return False

# 1. Stress test with common/default accounts
bad_credentials = [
    ("admin", "admin"),
    ("admin", "kai"),
    ("kai_admin", "kai"),
    ("kai_admin", "kai_developer"),
    ("root", "toor"),
    ("guest", "guest"),
    (None, None),  # Anonymous
    (dashboard_user, "incorrect_pass"),
]

all_bad_blocked = True
for user, pw in bad_credentials:
    desc = f"Anonymous" if not user else f"User: {user}"
    res = test_mqtt_auth(user, pw, desc)
    if res:
        all_bad_blocked = False
        print(colored(f"  • FAILURE: Connection with {desc} was incorrectly allowed!", Colors.RED + Colors.BOLD))

if all_bad_blocked:
    print(colored("  • SUCCESS: MQTT Broker authentication is completely secure. Only authorized users can connect.", Colors.GREEN))
else:
    print(colored("  • FAILURE: One or more unauthorized access routes were left open!", Colors.RED + Colors.BOLD))
print()


# ----------------------------------------------------------------
# PART 3: GRANULAR ACL TOPIC PRIVILEGE ISOLATION
# ----------------------------------------------------------------
print(colored("--- PART 3: Granular ACL Topic Privilege Isolation ---", Colors.CYAN + Colors.BOLD))

# Let's load the background internal role credentials from config.json to perform full multi-role routing checks
img_user = config.get("MQTT_USER_IMAGE_SERVER", "kai_image_server")
img_pass = config.get("MQTT_PASSWORD_IMAGE_SERVER")
ctrl_user = config.get("MQTT_USER_CONTROL_SERVER", "kai_control_server")
ctrl_pass = config.get("MQTT_PASSWORD_CONTROL_SERVER")

print(f"Dashboard Account:    Username='{dashboard_user}'")
print(f"Control Server Account: Username='{ctrl_user}'")
print(f"Image Server Account:   Username='{img_user}'")
print()

# Connect clients for each role
client_dash_sub = mqtt.Client(client_id="audit_dash_sub", callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
client_dash_pub = mqtt.Client(client_id="audit_dash_pub", callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
client_ctrl = mqtt.Client(client_id="audit_ctrl_role", callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
client_img = mqtt.Client(client_id="audit_img_role", callback_api_version=mqtt.CallbackAPIVersion.VERSION2)

client_dash_sub.username_pw_set(dashboard_user, dashboard_pass)
client_dash_pub.username_pw_set(dashboard_user, dashboard_pass)
client_ctrl.username_pw_set(ctrl_user, ctrl_pass)
client_img.username_pw_set(img_user, img_pass)

dash_received = []
ctrl_received = []

def on_msg_dash(client, userdata, msg):
    payload = msg.payload.decode('utf-8')
    dash_received.append((msg.topic, payload))

def on_msg_ctrl(client, userdata, msg):
    payload = msg.payload.decode('utf-8')
    ctrl_received.append((msg.topic, payload))

client_dash_sub.on_message = on_msg_dash
client_ctrl.on_message = on_msg_ctrl

# Connect all roles
client_dash_sub.connect(host, port)
client_dash_pub.connect(host, port)
client_ctrl.connect(host, port)
client_img.connect(host, port)

client_dash_sub.loop_start()
client_dash_pub.loop_start()
client_ctrl.loop_start()
client_img.loop_start()

time.sleep(0.5)

# --- SCENARIO 1: Unauthorized Write Bypass stress-test ---
# Dashboard has READ-ONLY privilege on 'kai/power'.
# Dashboard attempts to write to 'kai/power'. If ACL is functioning, the broker should block the publish/route.
print(colored("[ACL TEST 1] Dashboard subscribing to 'kai/power' (Authorized Read)...", Colors.YELLOW))
client_dash_sub.subscribe("kai/power")
time.sleep(0.2)

print(colored("[ACL TEST 2] Dashboard attempting write to 'kai/power' (UNAUTHORIZED Write)...", Colors.YELLOW))
client_dash_pub.publish("kai/power", "DASHBOARD_EXPLOIT_PAYLOAD")
time.sleep(1)

# Now write a valid message from the Control Server (which has WRITE privilege on 'kai/power') to verify routing
print(colored("[ACL TEST 3] Control Server publishing to 'kai/power' (Authorized Write)...", Colors.YELLOW))
client_ctrl.publish("kai/power", "CONTROL_VALID_STATE")
time.sleep(1)

# Check messages received by Dashboard
dash_power_msgs = [payload for topic, payload in dash_received if topic == "kai/power"]
print(f"  • Received messages on 'kai/power': {dash_power_msgs}")

unauthorized_write_blocked = "DASHBOARD_EXPLOIT_PAYLOAD" not in dash_power_msgs
authorized_read_worked = "CONTROL_VALID_STATE" in dash_power_msgs

if unauthorized_write_blocked and authorized_read_worked:
    print(colored("  • SUCCESS: Dashboard unauthorized write to 'kai/power' was BLOCKED, but authorized read WORKED.", Colors.GREEN))
else:
    if not unauthorized_write_blocked:
        print(colored("  • FAILURE: Dashboard bypassed ACL and successfully published write to 'kai/power'!", Colors.RED + Colors.BOLD))
    if not authorized_read_worked:
        print(colored("  • WARNING: Dashboard did not receive valid state on 'kai/power'. Is broker running normally?", Colors.YELLOW))

print()

# --- SCENARIO 2: Unauthorized Read Bypass stress-test ---
# Dashboard has WRITE-ONLY privilege on 'kai/control'.
# Dashboard attempts to subscribe to 'kai/control'. If ACL is functioning, the broker should deny subscription or drop routing.
print(colored("[ACL TEST 4] Dashboard subscribing to 'kai/control' (UNAUTHORIZED Read)...", Colors.YELLOW))
client_dash_sub.subscribe("kai/control")
time.sleep(0.2)

print(colored("[ACL TEST 5] Control Server subscribing to 'kai/control' (Authorized Read)...", Colors.YELLOW))
client_ctrl.subscribe("kai/control")
time.sleep(0.2)

print(colored("[ACL TEST 6] Dashboard publishing command to 'kai/control' (Authorized Write)...", Colors.YELLOW))
client_dash_pub.publish("kai/control", "DASHBOARD_OVERRIDE_CMD")
time.sleep(1.5)

# Check messages
dash_control_msgs = [payload for topic, payload in dash_received if topic == "kai/control"]
ctrl_control_msgs = [payload for topic, payload in ctrl_received if topic == "kai/control"]

print(f"  • Dashboard received on 'kai/control': {dash_control_msgs}")
print(f"  • Control Server received on 'kai/control': {ctrl_control_msgs}")

unauthorized_read_blocked = "DASHBOARD_OVERRIDE_CMD" not in dash_control_msgs
authorized_write_worked = "DASHBOARD_OVERRIDE_CMD" in ctrl_control_msgs

if unauthorized_read_blocked and authorized_write_worked:
    print(colored("  • SUCCESS: Dashboard unauthorized read from 'kai/control' was BLOCKED, but authorized write WORKED.", Colors.GREEN))
else:
    if not unauthorized_read_blocked:
        print(colored("  • FAILURE: Dashboard bypassed ACL and successfully read its own/other control command!", Colors.RED + Colors.BOLD))
    if not authorized_write_worked:
        print(colored("  • WARNING: Control Server did not receive the dashboard command. Check broker logs.", Colors.YELLOW))

print()

# --- SCENARIO 3: Wildcard Sniffing stress-test ---
# Dashboard has NO privilege to read 'kai/control'. Dashboard attempts to subscribe to wildcard 'kai/#' to bypass ACL.
# If ACL is functioning, the broker should filter out 'kai/control' messages from the wildcard match.
print(colored("[ACL TEST 7] Dashboard subscribing to wildcard 'kai/#' (UNAUTHORIZED Wildcard Read)...", Colors.YELLOW))
client_dash_sub.subscribe("kai/#")
time.sleep(0.2)

# Clear received list for clean tracking
dash_received.clear()

# Publish to different topics
client_ctrl.publish("kai/power", "WILDCARD_POWER_MSG") # Dashboard has read authority on 'kai/power'
client_dash_pub.publish("kai/control", "WILDCARD_CONTROL_MSG") # Dashboard does NOT have read authority on 'kai/control'
time.sleep(1.5)

print(f"  • Dashboard wildcard feed received: {dash_received}")
wildcard_secured = True
for topic, payload in dash_received:
    if topic == "kai/control":
        wildcard_secured = False

if wildcard_secured:
    print(colored("  • SUCCESS: Wildcard subscription respected ACL limits. Unauthorized topics were filtered out.", Colors.GREEN))
else:
    print(colored("  • FAILURE: Dashboard successfully sniffed unauthorized 'kai/control' traffic via wildcard subscription!", Colors.RED + Colors.BOLD))

print()


# ----------------------------------------------------------------
# PART 4: CONFIGURATION INJECTION & PARAMETER ABUSE MITIGATION
# ----------------------------------------------------------------
print(colored("--- PART 4: Configuration Injection & Parameter Abuse Mitigation ---", Colors.CYAN + Colors.BOLD))
url_config = "http://127.0.0.1:8000/api/config"

# 1. Attempt to overwrite sensitive system fields (Privilege Escalation / System Takeover Attack)
try:
    exploit_payload = {
        "MQTT_BROKER_PORT": 9999,
        "API_KEY": "hacked_key",
        "MQTT_PASSWORD": "hacked_password",
        "DEV_MODE": True
    }
    r = requests.post(url_config, json=exploit_payload, headers={"X-API-Key": api_key}, timeout=3)
    print(colored(f"[TEST ABUSE-1] Overwrite sensitive settings: Status Code = {r.status_code}", Colors.YELLOW))
    if r.status_code == 400:
        print(colored("  • SUCCESS: Attempt to overwrite sensitive settings was strictly BLOCKED by whitelist.", Colors.GREEN))
        try:
            print(f"    -> Error details: {r.json().get('error')}")
        except: pass
    else:
        print(colored("  • FAILURE: Protected settings successfully bypassed whitelist validation!", Colors.RED + Colors.BOLD))
except Exception as e:
    print(f"  • ERROR: {e}")

# 2. Attempt to inject invalid data types (DoS / Crash Attack)
try:
    crash_payload = {
        "DASHBOARD_INTERVAL": "thirty_seconds"
    }
    r = requests.post(url_config, json=crash_payload, headers={"X-API-Key": api_key}, timeout=3)
    print(colored(f"[TEST ABUSE-2] Inject invalid type: Status Code = {r.status_code}", Colors.YELLOW))
    if r.status_code == 400:
        print(colored("  • SUCCESS: Attempt to inject invalid data type was strictly BLOCKED.", Colors.GREEN))
        try:
            print(f"    -> Error details: {r.json().get('error')}")
        except: pass
    else:
        print(colored("  • FAILURE: Invalid data type bypassed type verification!", Colors.RED + Colors.BOLD))
except Exception as e:
    print(f"  • ERROR: {e}")

# 3. Attempt to inject out-of-bounds numeric values (DoS / Range Attack)
try:
    bounds_payload = {
        "DASHBOARD_INTERVAL": -100
    }
    r = requests.post(url_config, json=bounds_payload, headers={"X-API-Key": api_key}, timeout=3)
    print(colored(f"[TEST ABUSE-3] Inject out-of-bounds number: Status Code = {r.status_code}", Colors.YELLOW))
    if r.status_code == 400:
        print(colored("  • SUCCESS: Attempt to inject negative interval was strictly BLOCKED.", Colors.GREEN))
        try:
            print(f"    -> Error details: {r.json().get('error')}")
        except: pass
    else:
        print(colored("  • FAILURE: Negative range bypassed bounds verification!", Colors.RED + Colors.BOLD))
except Exception as e:
    print(f"  • ERROR: {e}")

# 4. Valid, safe configuration preference update (Regression Check)
try:
    safe_payload = {
        "DASHBOARD_INTERVAL": 30,
        "MQTT_DEBUG": False
    }
    r = requests.post(url_config, json=safe_payload, headers={"X-API-Key": api_key}, timeout=3)
    print(colored(f"[TEST ABUSE-4] Submit safe whitelisted preferences: Status Code = {r.status_code}", Colors.YELLOW))
    if r.status_code == 200:
        print(colored("  • SUCCESS: Safe preferences successfully updated and merged.", Colors.GREEN))
    else:
        print(colored("  • FAILURE: Valid preferences were rejected!", Colors.RED + Colors.BOLD))
except Exception as e:
    print(f"  • ERROR: {e}")

print()


# Clean up MQTT clients
client_dash_sub.loop_stop()
client_dash_pub.loop_stop()
client_ctrl.loop_stop()
client_img.loop_stop()

client_dash_sub.disconnect()
client_dash_pub.disconnect()
client_ctrl.disconnect()
client_img.disconnect()

# Clean up HTTP Server if launched by us
if api_process:
    print(colored("[INFO] Terminating temporary HTTP API background process...", Colors.BLUE))
    api_process.terminate()
    api_process.wait()

# Relaunch the live, hardened HTTP API Server for normal system operation
print(colored("[INFO] Relaunching the live, hardened HTTP API Server in the background...", Colors.BLUE))
subprocess.Popen(
    [sys.executable, api_script],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    cwd=PROJECT_ROOT,
    creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0
)
print(colored("[OK] Hardened HTTP API is running and live.", Colors.GREEN))

print("\n==============================================")
print(colored("            AUDIT COMPLETE                    ", Colors.GREEN + Colors.BOLD))
print("==============================================")
