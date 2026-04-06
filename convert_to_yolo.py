import json
import os
import glob
import random
import shutil
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
JSON_DIR   = r"C:\xampp\htdocs\python_test\bottle"
OUTPUT_DIR = r"C:\xampp\htdocs\python_test\bottle\YOLODataset"

# ✅ CORRECT - only 3 classes with underscore
CLASS_NAMES = [
    "good_bottle",
    "broken_glass",
    "dirty_bottle"
]
# ─────────────────────────────────────────────

def convert_labelme_to_yolo(json_path, output_label_path, class_names):
    with open(json_path, "r") as f:
        data = json.load(f)

    img_width  = data["imageWidth"]
    img_height = data["imageHeight"]
    lines = []

    for shape in data["shapes"]:
        label = shape["label"]

        # Auto-fix common label variations
        label_lower = label.lower().replace(" ", "_")
        if "good" in label_lower:
            label = "good_bottle"
        elif "broken" in label_lower:
            label = "broken_glass"
        elif "dirty" in label_lower:
            label = "dirty_bottle"

        if label not in class_names:
            print(f"  ⚠️  Unknown label '{label}' — skipping")
            continue

        class_id = class_names.index(label)
        points   = shape["points"]

        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        # Convert to YOLO normalized format
        x_center = ((x_min + x_max) / 2) / img_width
        y_center = ((y_min + y_max) / 2) / img_height
        width    = (x_max - x_min) / img_width
        height   = (y_max - y_min) / img_height

        # Clamp to valid range
        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))
        width    = max(0.0, min(1.0, width))
        height   = max(0.0, min(1.0, height))

        lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    with open(output_label_path, "w") as f:
        f.write("\n".join(lines))

def main():
    # Create output folders
    for split in ["train", "val"]:
        os.makedirs(f"{OUTPUT_DIR}/{split}/images", exist_ok=True)
        os.makedirs(f"{OUTPUT_DIR}/{split}/labels", exist_ok=True)

    # Find all JSON files
    json_files = glob.glob(os.path.join(JSON_DIR, "*.json"))
    print(f"Found {len(json_files)} labeled images")

    if len(json_files) == 0:
        print("❌ No JSON files found!")
        return

    # ✅ Shuffle so all classes spread evenly across train/val
    random.shuffle(json_files)
    split_idx   = int(len(json_files) * 0.8)
    train_files = json_files[:split_idx]
    val_files   = json_files[split_idx:]

    for i, json_path in enumerate(json_files):
        split = "train" if i < split_idx else "val"
        stem  = Path(json_path).stem

        # Copy image
        for ext in [".jpg", ".jpeg", ".png"]:
            img_src = os.path.join(JSON_DIR, stem + ext)
            if os.path.exists(img_src):
                shutil.copy(img_src, f"{OUTPUT_DIR}/{split}/images/{stem}{ext}")
                break

        # Convert label
        label_dst = f"{OUTPUT_DIR}/{split}/labels/{stem}.txt"
        convert_labelme_to_yolo(json_path, label_dst, CLASS_NAMES)
        print(f"  ✅ [{split}] {stem}")

    # ✅ Create correct data.yaml with only 3 classes
    yaml_content = f"""path: {OUTPUT_DIR}
train: train/images
val: val/images

nc: 3
names:
  - good_bottle
  - broken_glass
  - dirty_bottle
"""
    with open(f"{OUTPUT_DIR}/data.yaml", "w") as f:
        f.write(yaml_content)

    print(f"\n✅ Done!")
    print(f"   Train: {len(train_files)} images")
    print(f"   Val:   {len(val_files)} images")
    print(f"   Saved to: {OUTPUT_DIR}")
    print(f"\n▶ Now run: py -3.11 train_model.py")

if __name__ == "__main__":
    main()