import cv2
import json
import os
import glob
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
IMAGE_FOLDER = r"C:\xampp\htdocs\python_test\bottle"
CLASS_NAME = "dirty_bottle"  # change per class
# ─────────────────────────────────────────────

images = glob.glob(os.path.join(IMAGE_FOLDER, "*.jpg"))
count  = 0

print(f"Found {len(images)} images")
print("Draw box around bottle | ENTER=save | S=skip | Q=quit")

for img_path in images:
    json_path = str(Path(img_path).with_suffix(".json"))

    # Skip already labeled
    if os.path.exists(json_path):
        print(f"  ⏭️  Already labeled: {Path(img_path).name}")
        continue

    img = cv2.imread(img_path)
    img = cv2.resize(img, (1280, 720))
    h, w = img.shape[:2]

    box = [0, 0, 0, 0]
    drawing = False
    start_x = start_y = 0

    clone = img.copy()

    def draw_box(event, x, y, flags, param):
        global drawing, start_x, start_y, box, img
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            start_x, start_y = x, y
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            img = clone.copy()
            cv2.rectangle(img, (start_x, start_y), (x, y), (0, 255, 0), 2)
            box = [start_x, start_y, x, y]
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            box = [start_x, start_y, x, y]

    cv2.namedWindow("Label Image")
    cv2.setMouseCallback("Label Image", draw_box)

    cv2.putText(img, f"Label: {CLASS_NAME}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(img, "Draw box | ENTER=save | S=skip | Q=quit", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    while True:
        cv2.putText(img, f"Label: {CLASS_NAME}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(img, "Draw box | ENTER=save | S=skip | Q=quit", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("Label Image", img)
        key = cv2.waitKey(1) & 0xFF

        if key == 13:  # ENTER — save label
            if box[2] > 0 and box[3] > 0:
                # Save as LabelMe JSON
                label_data = {
                    "version": "5.0.0",
                    "flags": {},
                    "shapes": [{
                        "label": CLASS_NAME,
                        "points": [
                            [box[0], box[1]],
                            [box[2], box[3]]
                        ],
                        "group_id": None,
                        "shape_type": "rectangle",
                        "flags": {}
                    }],
                    "imagePath": Path(img_path).name,
                    "imageHeight": h,
                    "imageWidth": w
                }
                with open(json_path, "w") as f:
                    json.dump(label_data, f, indent=2)
                count += 1
                print(f"  ✅ Saved [{count}]: {Path(img_path).name}")
            break

        elif key == ord('s'):  # skip
            print(f"  ⏭️  Skipped: {Path(img_path).name}")
            break

        elif key == ord('q'):  # quit
            cv2.destroyAllWindows()
            print(f"\n✅ Done! Labeled {count} images")
            exit()

cv2.destroyAllWindows()
print(f"\n✅ Done! Labeled {count} images")
print("Now run: python convert_to_yolo.py")