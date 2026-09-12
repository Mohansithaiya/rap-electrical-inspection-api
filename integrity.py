import os
import json
import hashlib
from collections import defaultdict

splits = ["train", "valid", "test"]

all_hashes = defaultdict(list)

for split in splits:

    folder = split
    annotation_file = os.path.join(
        folder, "_annotations.coco.json"
    )

    with open(annotation_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    image_files = {
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    }

    coco_images = {
        img["file_name"]
        for img in data["images"]
    }

    missing = coco_images - image_files
    unused = image_files - coco_images

    print(f"\n===== {split.upper()} =====")
    print("Images on disk:", len(image_files))
    print("Images in COCO:", len(coco_images))
    print("Missing files:", len(missing))
    print("Unused images:", len(unused))

    for filename in image_files:
        path = os.path.join(folder, filename)

        with open(path, "rb") as f:
            file_hash = hashlib.md5(f.read()).hexdigest()

        all_hashes[file_hash].append(
            (split, filename)
        )

duplicates = {
    h: files
    for h, files in all_hashes.items()
    if len(files) > 1
}

print("\n===== DUPLICATE CHECK =====")
print("Duplicate image groups:", len(duplicates))

cross_split = 0

for files in duplicates.values():
    split_names = set(x[0] for x in files)

    if len(split_names) > 1:
        cross_split += 1
        print("CROSS-SPLIT DUPLICATE:", files)

print("Cross-split duplicate groups:", cross_split)