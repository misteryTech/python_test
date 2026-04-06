import cv2
from ultralytics import YOLO

# Load a free pretrained model (downloads automatically ~6MB)
model = YOLO("yolov8n.pt")

# Open camera (change 0 to 1 if no camera found)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("No camera found! Try changing index to 1 or 2")
    exit()

print("Camera opened! Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)
    frame = results[0].plot()

    frame = cv2.resize(frame, (1280, 720))  # ✅ moved here, inside the loop

    cv2.imshow("Test Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()