import glob
import os

FOLDERS = [
    r"C:\xampp\htdocs\python_test\bottle\YOLODataset\train\labels",
    r"C:\xampp\htdocs\python_test\bottle\YOLODataset\val\labels",
]

fixed = 0
errors = 0

for folder in FOLDERS:
    for file_path in glob.glob(os.path.join(folder, "*.txt")):
        with open(file_path, "r") as f:
            lines = f.readlines()

        new_lines = []
        changed = False
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 5:
                cls = parts[0]
                x, y, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

                # Clamp all values between 0 and 1
                x = max(0.0, min(1.0, x))
                y = max(0.0, min(1.0, y))
                w = max(0.0, min(1.0, w))
                h = max(0.0, min(1.0, h))

                new_line = f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
                if new_line != line.strip():
                    changed = True
                new_lines.append(new_line)
            else:
                errors += 1

        if changed:
            with open(file_path, "w") as f:
                f.write("\n".join(new_lines))
            fixed += 1

print(f"✅ Fixed {fixed} files")
print(f"⚠️  Skipped {errors} bad lines")
print("Now run: py -3.11 train_model.py")