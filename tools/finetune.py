import os
from ultralytics import YOLO
import torch

# Resolve paths from the project root so results land where image_server.py looks for them,
# regardless of the directory this script is launched from.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def finetune_model():
    """
    This script fine-tunes a YOLOv11 model on a custom dataset.

    Before running, make sure your `dataset.yaml` file is correctly configured
    with the paths to your training and validation data.
    """
    # --- Configuration ---
    DATASET_CONFIG = 'dataset.yaml'
    EPOCHS = 50  # Number of training epochs
    BATCH_SIZE = 8 # Adjust based on your GPU memory
    MODEL_TO_FINETUNE = os.path.join(PROJECT_ROOT, 'models', 'yolo11s.pt') # Same base model as image_server.py

    # --- Device Selection ---
    def get_device():
        """Check for CUDA device and return it, otherwise fallback to CPU."""
        if torch.cuda.is_available():
            print("[Fine-tuning] CUDA is available. Using GPU.")
            return 'cuda'
        print("[Fine-tuning] CUDA not available. Using CPU.")
        return 'cpu'

    DEVICE = get_device()
    
    # --- Load Model ---
    try:
        model = YOLO(MODEL_TO_FINETUNE)
        model.to(DEVICE)
        print(f"[Fine-tuning] Loaded model: {MODEL_TO_FINETUNE}")
    except Exception as e:
        print(f"[Error] Could not load model: {e}")
        return

    # --- Start Fine-Tuning ---
    print("[Fine-tuning] Starting fine-tuning process...")
    try:
        model.train(
            data=DATASET_CONFIG,
            epochs=EPOCHS,
            batch=BATCH_SIZE,
            imgsz=640, # Image size
            project=os.path.join(PROJECT_ROOT, 'runs', 'train'),
            name='yolo_finetune_results'
        )
        print("[Fine-tuning] Training complete.")

        # The best model is automatically saved in `runs/train/yolo_finetune_results/weights/best.pt`
        print(f"Best model saved in 'runs/train/yolo_finetune_results/weights/best.pt'")
        print(f"image_server.py picks it up automatically on next start.")

    except Exception as e:
        print(f"[Error] An error occurred during training: {e}")

if __name__ == '__main__':
    finetune_model()
