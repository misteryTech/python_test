import cv2
from ultralytics import YOLO

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
MODEL_PATH = "runs/detect/bottle_qc_model/weights/best.pt"
IMAGE_PATH = r"C:\xampp\htdocs\python_test\bottle\good_bottle_20260329_001924_056850.jpg"
CONFIDENCE_THRESHOLD = 0.1  # very low to catch anything
# ─────────────────────────────────────────────

model = YOLO(MODEL_PATH)

# Run detection on image
results = model(IMAGE_PATH, conf=0.01)  # lowest possible

# Print all detections with confidence
print("\n─────────────────────────────")
print("  DETECTION RESULTS")
print("─────────────────────────────")

for result in results:
    if len(result.boxes) == 0:
        print("  ❌ Nothing detected")
    for box in result.boxes:
        cls_id     = int(box.cls[0])
        label      = result.names[cls_id]
        confidence = float(box.conf[0])
        print(f"  ✅ {label}: {confidence:.1%} confidence")

# Show image with detections
annotated = results[0].plot()
cv2.imshow("Test Image Result", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()