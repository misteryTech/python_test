import glob
import os

FOLDERS = [
    r"C:\xampp\htdocs\python_test\bottle\YOLODataset\train\labels",
    r"C:\xampp\htdocs\python_test\bottle\YOLODataset\val\labels",
]

# Map everything to correct IDs based on filename
for folder in FOLDERS:
    fixed = 0
    for file_path in glob.glob(os.path.join(folder, "*.txt")):
        name = os.path.basename(file_path).lower()

        # Determine correct class from filename
        if "good" in name:
            correct_id = "0"
        elif "broken" in name:
            correct_id = "1"
        elif "dirty" in name:
            correct_id = "2"
        else:
            correct_id = "0"  # default

        with open(file_path, "r") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 5:
                parts[0] = correct_id
                new_lines.append(" ".join(parts))

        with open(file_path, "w") as f:
            f.write("\n".join(new_lines))
        fixed += 1

    print(f"✅ Fixed {fixed} files in {folder}")

print("\n✅ Done! Now run: py -3.11 check_ids.py")