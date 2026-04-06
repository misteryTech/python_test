import glob

folders = [
    r"C:\xampp\htdocs\python_test\bottle\YOLODataset\train\labels",
    r"C:\xampp\htdocs\python_test\bottle\YOLODataset\val\labels",
]

for folder in folders:
    ids = set()
    for f in glob.glob(folder + "/*.txt"):
        with open(f) as r:
            for line in r:
                p = line.strip().split()
                if p:
                    ids.add(p[0])
    print(folder[-15:], "IDs:", sorted(ids))