import os
import time
import random
import shutil
from typing import Dict, Iterable, List

import torch
from ultralytics import YOLO

from feeds import append_message, consume_messages

IMAGE_SRC_DIR = "images_src"
IMAGE_READY_DIR = "images_ready"
FORCE_REQUEST_FEED = "force_request"
FORCE_SERVED_FEED = "force_served"
CAPTURE_INTERVAL = 60  # seconds
BATCH_SIZE = 25
YOLO_MODEL_NAME = 'yolo11s.pt'  # Change this to use a different YOLO model

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
    FINETUNED_MODEL_PATH = 'runs/train/yolo_finetune_results/weights/best.pt'
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


def detect_person_in_batch(image_paths: Iterable[str]) -> Dict[str, str]:
    """
    Runs person detection on a batch of images using YOLOv8.
    Returns a dictionary mapping image path to 'YES' or 'NO'.
    """
    if not MODEL:
        print("[YOLO] Model not loaded, returning random status.")
        return {path: ("YES" if random.random() > 0.5 else "NO") for path in image_paths}

    image_paths = list(image_paths)
    results_map: Dict[str, str] = {path: "NO" for path in image_paths}
    try:
        # Process images in batches
        predictions = MODEL.predict(source=image_paths, device=DEVICE, classes=[0], verbose=False) # Class 0 is 'person'
        
        for i, result in enumerate(predictions):
            if len(result.boxes) > 0:  # A person was detected
                results_map[image_paths[i]] = "YES"

    except Exception as e:
        print(f"[YOLO] Error during batch prediction: {e}")
    
    return results_map


def _camera_id_from_path(image_path: str) -> str:
    """Extract camera ID from filename (format: CAM_anynamehere.jpg)"""
    filename = os.path.basename(image_path)
    stem = os.path.splitext(filename)[0]
    
    # Check if it starts with CAM_
    if stem.startswith("CAM_"):
        return stem  # Return full name including CAM_ prefix
    
    # Fallback for files not following the convention
    return stem


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
        parsed = _parse_ready_filename(entry)
        if parsed and parsed[0] == camera_id:
            try:
                os.remove(os.path.join(IMAGE_READY_DIR, entry))
            except OSError as exc:
                print(f"[Image Server] Unable to remove old file '{entry}': {exc}")


def _write_ready_image(source_path: str, camera_id: str, status: str) -> None:
    _, ext = os.path.splitext(source_path)
    ext = ext.lower() or ".jpg"
    if ext not in {".jpg", ".jpeg", ".png"}:
        ext = ".jpg"
    destination_name = f"{camera_id}_{status}.jpg"
    destination_path = os.path.join(IMAGE_READY_DIR, destination_name)

    _remove_existing_ready_file(camera_id)

    try:
        shutil.copy2(source_path, destination_path)  # Changed from shutil.move to shutil.copy2
    except Exception as exc:
        print(f"[Image Server] Failed to copy '{source_path}' to ready dir: {exc}")


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
    if not all_image_files:
        return

    # Process in batches
    for i in range(0, len(all_image_files), BATCH_SIZE):
        batch_paths = all_image_files[i:i + BATCH_SIZE]
        print(f"[Image Server] Processing batch {i//BATCH_SIZE + 1} with {len(batch_paths)} images.")
        
        start_time = time.time()
        detection_results = detect_person_in_batch(batch_paths)
        end_time = time.time()

        print(f"[Image Server] Batch processed in {end_time - start_time:.2f} seconds.")

        # Create new files in 'ready', keep originals in 'src'
        for img_path, status in detection_results.items():
            camera_id = _camera_id_from_path(img_path)
            status_token = status.upper()
            if status_token not in {"YES", "NO"}:
                status_token = "NO"
            _write_ready_image(img_path, camera_id, status_token)
            print(f"[Image Server] Updated camera '{camera_id}' with status {status_token}.")


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
        status = detection_result.get(source_image_path, "NO").upper()
        if status not in {"YES", "NO"}:
            status = "NO"

        _write_ready_image(source_image_path, cam_id, status)
        append_message(FORCE_SERVED_FEED, f"UPDATED_{cam_id}")
        print(f"[Image Server] Served force update for {cam_id}, status: {status}")


def main():
    """Main loop for the image server."""
    print("Image Server started.")
    os.makedirs(IMAGE_SRC_DIR, exist_ok=True)
    os.makedirs(IMAGE_READY_DIR, exist_ok=True)

    last_capture_time = 0
    while True:
        # Process force requests immediately
        process_force_requests()

        # Regular capture every CAPTURE_INTERVAL seconds
        current_time = time.time()
        if current_time - last_capture_time >= CAPTURE_INTERVAL:
            capture_and_update_images()
            last_capture_time = current_time

        time.sleep(1)

if __name__ == "__main__":
    main()
