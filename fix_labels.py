import glob
import os

FOLDERS = [
    r"C:\xampp\htdocs\python_test\bottle\YOLODataset\train\labels",
    r"C:\xampp\htdocs\python_test\bottle\YOLODataset\val\labels",
]

# Force remap ALL class IDs to valid 0,1,2
REMAP = {
    "0": "0",
    "1": "0",
    "2": "1",
    "3": "2",
    "4": "0",
    "5": "0",
}

VALID = {"0", "1", "2"}

for folder in FOLDERS:
    files = glob.glob(os.path.join(folder, "*.txt"))
    fixed = 0
    deleted = 0
    for file_path in files:
        with open(file_path, "r") as f:
            lines = f.readlines()

        new_lines = []
        changed = False
        skip = False
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            cls_id = parts[0]

            # Remap if needed
            if cls_id in REMAP:
                new_cls = REMAP[cls_id]
            else:
                print(f"  ⚠️ Unknown ID {cls_id} in {file_path} — skipping line")
                skip = True
                continue

            if cls_id != new_cls:
                changed = True
            parts[0] = new_cls
            new_lines.append(" ".join(parts))

        if new_lines:
            with open(file_path, "w") as f:
                f.write("\n".join(new_lines))
            if changed:
                fixed += 1
        else:
            # Empty file - delete it
            os.remove(file_path)
            deleted += 1
            print(f"  🗑️ Deleted empty: {os.path.basename(file_path)}")

    print(f"✅ Fixed {fixed} files, deleted {deleted} empty files in:")
    print(f"   {folder}")

print("\n✅ All done! Now run: py -3.11 train_model.py")