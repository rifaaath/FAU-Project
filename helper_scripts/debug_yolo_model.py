# debug_yolo_model.py
from ultralytics import YOLO

# --- CONFIGURE YOUR MODEL PATH ---
MODEL_PATH = 'yolo_training_run/train/weights/best.pt'
# --------------------------------

print(f"Loading YOLO model from: {MODEL_PATH}")
model = YOLO(MODEL_PATH)

print("\n--- Authoritative Class Map from Inside the Model ---")
# model.names is a dictionary like {0: 'alpha', 1: 'beta', ...}
class_map = model.names
for class_id, class_name in sorted(class_map.items()):
    print(f"  Category ID: {class_id} -> Name: '{class_name}'")

print(f"\nThis is the ground truth. When the model predicts ID 7, it means '{class_map.get(7)}'.")