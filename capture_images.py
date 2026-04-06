import cv2
import os
from datetime import datetime

# ─────────────────────────────────────────────
# CHANGE THIS EACH TIME YOU CAPTURE A NEW CLASS
# ─────────────────────────────────────────────
CLASS_NAME = "broken_glass"  # change to:
                              # "broken_glass"
                              # "dirty_bottle"
SAVE_FOLDER = r"C:\xampp\htdocs\python_test\bottle"
# ─────────────────────────────────────────────

os.makedirs(f"{SAVE_FOLDER}", exist_ok=True)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

count = 0
print(f"📸 Capturing: {CLASS_NAME}")
print("Press SPACE to capture | Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1280, 720))

    cv2.putText(frame, f"Class: {CLASS_NAME}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Captured: {count} images", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, "SPACE = capture | Q = quit", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Capture Bottle Images", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{SAVE_FOLDER}/{CLASS_NAME}_{ts}.jpg"
        cv2.imwrite(filename, frame)
        count += 1
        print(f"  ✅ Saved {count}: {filename}")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"\n✅ Done! {count} images saved!")