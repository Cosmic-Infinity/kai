import os
import sys
import time
import shutil
from typing import Dict, Iterable, List

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import json
from datetime import datetime
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO

from feeds import append_message, consume_messages

# Use absolute paths based on project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_SRC_DIR = os.path.join(PROJECT_ROOT, "images_src")
IMAGE_READY_DIR = os.path.join(PROJECT_ROOT, "images_ready")
FORCE_REQUEST_FEED = "force_request"
FORCE_SERVED_FEED = "force_served"
BATCH_SIZE = 25
YOLO_MODEL_NAME = os.path.join(PROJECT_ROOT, 'models', 'yolo11s.pt')  # Change this to use a different YOLO model
from config_manager import load_config, load_frame_log, save_frame_log
from logger import kai_print as print

# --- YOLO Model Initialization ---
def get_device():
    """Check for CUDA device and return it, otherwise fallback to CPU."""
    if torch.cuda.is_available():
        print("[YOLO] CUDA is available. Using GPU.")
        return 'cuda'
    print("[YOLO] CUDA not available. Using CPU.")
    return 'cpu'

DEVICE = get_device()
try:
    # Use the fine-tuned model if available, otherwise fall back to the base model
    FINETUNED_MODEL_PATH = os.path.join(PROJECT_ROOT, 'runs/train/yolo_finetune_results/weights/best.pt')
    if os.path.exists(FINETUNED_MODEL_PATH):
        MODEL = YOLO(FINETUNED_MODEL_PATH)
        print(f"[YOLO] Loaded fine-tuned model from: {FINETUNED_MODEL_PATH}")
    else:
        MODEL = YOLO(YOLO_MODEL_NAME)
        print(f"[YOLO] Fine-tuned model not found. Loaded base '{YOLO_MODEL_NAME}' model.")
    
    MODEL.to(DEVICE)
except Exception as e:
    print(f"[YOLO] Error loading model: {e}")
    MODEL = None
# --- End YOLO Initialization ---


def _result_to_detection(result) -> tuple:
    """Convert one YOLO result into ('YES' or 'NO', bboxes_list)."""
    bboxes = []
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxyn[0].tolist()
        conf = float(box.conf[0])
        # Map to UI format (fx, fy, fw, fh) with bottom-left origin
        fx = x1
        fy = 1.0 - y2
        fw = x2 - x1
        fh = y2 - y1

        # Gradient color from Red (F43F5E) to Green (10B981) based on confidence
        r = int(244 + (16 - 244) * conf)
        g = int(63 + (185 - 63) * conf)
        b = int(94 + (129 - 94) * conf)
        color = f"#{r:02X}{g:02X}{b:02X}"

        bboxes.append((fx, fy, fw, fh, f"Person {int(conf*100)}%", color))
    return ("YES" if bboxes else "NO"), bboxes


def _predict(image_paths: List[str]):
    return MODEL.predict(source=image_paths, device=DEVICE, classes=[0], verbose=False)  # Class 0 is 'person'


def detect_person_in_batch(image_paths: Iterable[str]) -> Dict[str, tuple]:
    """
    Runs person detection on a batch of images using YOLOv11.
    Returns a dictionary mapping image path to ('YES' or 'NO', bboxes_list).

    Images that could not be processed are left out of the result (and the error is
    reported) so their previous status is kept - a failed detection must never be
    mistaken for an empty room, since 'NO' eventually powers appliances off.
    """
    image_paths = list(image_paths)
    if not MODEL:
        print("[YOLO] Model not loaded - skipping detection, existing camera statuses are left unchanged.")
        return {}

    try:
        predictions = _predict(image_paths)
        if len(predictions) != len(image_paths):
            raise RuntimeError(f"expected {len(image_paths)} results, got {len(predictions)}")
        pairs = list(zip(image_paths, predictions))
    except Exception as e:
        # Retry one by one so a single bad image doesn't block the rest of the batch
        print(f"[YOLO] Batch prediction failed ({e}); retrying images individually.")
        pairs = []
        for path in image_paths:
            try:
                pairs.append((path, _predict([path])[0]))
            except Exception as img_err:
                print(f"[YOLO] Skipping '{os.path.basename(path)}', status left unchanged: {img_err}")

    return {path: _result_to_detection(result) for path, result in pairs}


# --- Frame timekeeping ---
EXIF_DATETIME_ORIGINAL = 36867  # in the Exif sub-IFD
EXIF_DATETIME = 306
EXIF_SUB_IFD = 0x8769

# Camera clocks can't be trusted to be right, so they are only informational: every timing
# decision (staleness, inactivity) uses this server's clock via "processed_at".
CLOCK_SKEW_WARN_SECONDS = 300

_frame_log: Dict[str, dict] = load_frame_log()
_skewed_cameras = set()


def _exif_time(image_path: str):
    """The capture timestamp the camera wrote into the frame (camera clock, local time), or None."""
    try:
        with Image.open(image_path) as img:
            exif = img.getexif()
            stamp = exif.get_ifd(EXIF_SUB_IFD).get(EXIF_DATETIME_ORIGINAL) or exif.get(EXIF_DATETIME)
        if stamp:
            return datetime.strptime(str(stamp).strip(), "%Y:%m:%d %H:%M:%S").timestamp()
    except (OSError, UnidentifiedImageError, ValueError) as e:
        print(f"[Image Server] Could not read EXIF time from '{os.path.basename(image_path)}' ({e}); using file time.")
    return None


def _record_frame(camera_id: str, source_path: str) -> None:
    """
    Log when a camera's frame was taken and processed.

    captured_at: the camera's EXIF timestamp if present, otherwise the file's modification time
    (when the frame landed in images_src, by this server's clock). camera_clock_offset is how far
    the camera's clock disagrees with the file time; large offsets are reported once per camera.
    """
    entry = {"captured_at": None, "camera_clock_offset": None, "processed_at": time.time()}
    try:
        file_time = os.path.getmtime(source_path)
    except OSError as e:  # source frame replaced/removed since it was processed
        print(f"[Image Server] Could not read capture time for {camera_id}: {e}")
        _frame_log[camera_id] = entry
        return

    exif_time = _exif_time(source_path)
    entry["captured_at"] = exif_time if exif_time is not None else file_time
    if exif_time is not None:
        offset = exif_time - file_time
        entry["camera_clock_offset"] = round(offset, 1)
        if abs(offset) > CLOCK_SKEW_WARN_SECONDS:
            if camera_id not in _skewed_cameras:
                print(f"[Image Server] {camera_id}: camera clock is {offset:+.0f}s off from this server "
                      f"(EXIF vs file time). Its timestamps are informational only; check the camera's time settings.")
                _skewed_cameras.add(camera_id)
        else:
            _skewed_cameras.discard(camera_id)
    _frame_log[camera_id] = entry


def _save_frame_log() -> None:
    try:
        save_frame_log(_frame_log)
    except OSError as e:
        print(f"[Image Server] Could not persist frame timestamps: {e}")


def _camera_id_from_path(image_path: str) -> str:
    """Extract camera ID from filename (format: CAM_anynamehere.jpg)"""
    filename = os.path.basename(image_path)
    return os.path.splitext(filename)[0]


def _parse_ready_filename(filename: str):
    """Parse ready filename (format: CAM_anynamehere_YES.jpg or CAM_anynamehere_NO.jpg)"""
    stem, ext = os.path.splitext(filename)
    if ext.lower() not in {".jpg", ".jpeg", ".png"}:
        return None
    if "_" not in stem:
        return None
    
    # Split from the right to get the status (YES/NO)
    base, status = stem.rsplit("_", 1)
    status = status.upper()
    if status not in {"YES", "NO"}:
        return None
    
    # Verify base follows CAM_ format
    if not base.startswith("CAM_"):
        return None
    
    return base, status


def _remove_existing_ready_file(camera_id: str) -> None:
    if not os.path.exists(IMAGE_READY_DIR):
        return
    for entry in os.listdir(IMAGE_READY_DIR):
        if entry.endswith("_bboxes.json"):
            cid = entry.replace("_bboxes.json", "")
            if cid == camera_id:
                try:
                    os.remove(os.path.join(IMAGE_READY_DIR, entry))
                except OSError as exc:
                    print(f"[Image Server] Unable to remove old file '{entry}': {exc}")
        else:
            parsed = _parse_ready_filename(entry)
            if parsed and parsed[0] == camera_id:
                try:
                    os.remove(os.path.join(IMAGE_READY_DIR, entry))
                except OSError as exc:
                    print(f"[Image Server] Unable to remove old file '{entry}': {exc}")


def _cleanup_stale_ready_files(valid_camera_ids: set) -> None:
    for cid in [c for c in _frame_log if c not in valid_camera_ids]:
        del _frame_log[cid]
    if not os.path.exists(IMAGE_READY_DIR):
        return
    for entry in os.listdir(IMAGE_READY_DIR):
        if entry.endswith("_bboxes.json"):
            cid = entry.replace("_bboxes.json", "")
            if cid not in valid_camera_ids:
                try:
                    os.remove(os.path.join(IMAGE_READY_DIR, entry))
                except OSError as exc:
                    print(f"[Image Server] Unable to remove stale file '{entry}': {exc}")
        else:
            parsed = _parse_ready_filename(entry)
            if parsed:
                cid, _ = parsed
                if cid not in valid_camera_ids:
                    try:
                        os.remove(os.path.join(IMAGE_READY_DIR, entry))
                    except OSError as exc:
                        print(f"[Image Server] Unable to remove stale file '{entry}': {exc}")


def _write_ready_image(source_path: str, camera_id: str, status: str, bboxes: list) -> bool:
    """Publish a tagged frame to images_ready. Returns False (leaving the old files intact) on failure."""
    destination_name = f"{camera_id}_{status}.jpg"
    destination_path = os.path.join(IMAGE_READY_DIR, destination_name)
    bbox_path = os.path.join(IMAGE_READY_DIR, f"{camera_id}_bboxes.json")
    tmp_image_path = destination_path + ".tmp"

    try:
        # Stage the new frame first so a failed copy doesn't wipe the camera's current status
        shutil.copy2(source_path, tmp_image_path)
        _remove_existing_ready_file(camera_id)
        os.replace(tmp_image_path, destination_path)
        with open(bbox_path, 'w') as f:
            json.dump(bboxes, f)
        return True
    except OSError as exc:
        print(f"[Image Server] Failed to publish '{source_path}' to ready dir: {exc}")
        if os.path.exists(tmp_image_path):
            os.remove(tmp_image_path)
        return False


def _list_source_images() -> List[str]:
    """List all source images that follow the CAM_* naming convention"""
    try:
        all_files = []
        for entry in os.listdir(IMAGE_SRC_DIR):
            # Check file extension
            if not entry.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            
            # Check if filename starts with CAM_
            stem = os.path.splitext(entry)[0]
            if stem.startswith("CAM_"):
                all_files.append(os.path.join(IMAGE_SRC_DIR, entry))
            else:
                print(f"[Image Server] Skipping file '{entry}' - doesn't follow CAM_* naming convention")
        
        return all_files
    except FileNotFoundError:
        print(f"[Image Server] Source directory '{IMAGE_SRC_DIR}' not found or is empty.")
        return []


def capture_and_update_images():
    """
    Processes images from the images_src folder in batches, creates new status
    files in images_ready. Source images are preserved.
    """
    print("[Image Server] Checking for images to process...")
    os.makedirs(IMAGE_SRC_DIR, exist_ok=True)
    os.makedirs(IMAGE_READY_DIR, exist_ok=True)

    all_image_files = _list_source_images()
    
    # Cleanup stale files
    valid_cids = {_camera_id_from_path(p) for p in all_image_files}
    _cleanup_stale_ready_files(valid_cids)

    if not all_image_files:
        _save_frame_log()
        return

    # Process in batches
    for i in range(0, len(all_image_files), BATCH_SIZE):
        batch_paths = all_image_files[i:i + BATCH_SIZE]
        print(f"[Image Server] Processing batch {i//BATCH_SIZE + 1} with {len(batch_paths)} images.")
        
        start_time = time.time()
        detection_results = detect_person_in_batch(batch_paths)
        end_time = time.time()

        print(f"[Image Server] Batch processed dynamically in {end_time - start_time:.2f} seconds.")

        # Create new files in 'ready', keep originals in 'src'
        for img_path, (status, bboxes) in detection_results.items():
            camera_id = _camera_id_from_path(img_path)
            if _write_ready_image(img_path, camera_id, status, bboxes):
                _record_frame(camera_id, img_path)
                print(f"[Image Server] Updated camera '{camera_id}' with status {status}.")
        _save_frame_log()


def process_force_requests():
    """Processes force update requests from the dashboard for a single camera."""
    requests = consume_messages(FORCE_REQUEST_FEED)
    if not requests:
        return

    for req in requests:
        if not req.startswith("FORCE_UPDATE_"):
            print(f"[Image Server] Ignoring unrecognized request '{req}'.")
            continue

        cam_id = req[len("FORCE_UPDATE_") :].strip()
        if not cam_id:
            print(f"[Image Server] Received malformed force update request: '{req}'")
            continue
        
        # Ensure cam_id follows CAM_ format
        if not cam_id.startswith("CAM_"):
            print(f"[Image Server] Invalid camera ID format: '{cam_id}' (expected CAM_*)")
            continue

        print(f"[Image Server] Force update request for {cam_id}")

        # Find the corresponding source image in the source directory
        source_image_path = None
        for filename in os.listdir(IMAGE_SRC_DIR):
            stem, ext = os.path.splitext(filename)
            if ext.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            if stem == cam_id:
                source_image_path = os.path.join(IMAGE_SRC_DIR, filename)
                break

        if not source_image_path:
            print(f"[Image Server] Source image for {cam_id} not found in {IMAGE_SRC_DIR}.")
            continue

        detection_result = detect_person_in_batch([source_image_path])
        res = detection_result.get(source_image_path)
        if not res:
            print(f"[Image Server] Detection failed for {cam_id}; keeping its previous status and not signalling force_served.")
            continue
        status, bboxes = res

        if not _write_ready_image(source_image_path, cam_id, status, bboxes):
            continue
        _record_frame(cam_id, source_image_path)
        _save_frame_log()
        append_message(FORCE_SERVED_FEED, f"FORCE_SERVED_{cam_id}")
        print(f"[Image Server] Served force update for {cam_id}, status: {status}")


def main():
    """Main loop for the image server."""
    # Explicitly configure the module role for secure isolated authentication
    from feeds import configure_role
    configure_role("image_server")
    
    print("Image Server started.")
    os.makedirs(IMAGE_SRC_DIR, exist_ok=True)
    os.makedirs(IMAGE_READY_DIR, exist_ok=True)

    last_capture_time = 0
    while True:
        try:
            # Process force requests immediately
            process_force_requests()

            # Regular capture based on config
            current_time = time.time()
            config = load_config()
            capture_interval = config.get("IMAGE_SERVER_INTERVAL", 60)
            
            if current_time - last_capture_time >= capture_interval:
                capture_and_update_images()
                last_capture_time = current_time

            time.sleep(1)
        except Exception as e:
            print(f"[Image Server] Critical error in main loop: {e}")
            time.sleep(1)

if __name__ == "__main__":
    main()
