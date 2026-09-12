import json
import os
from collections import Counter

for split in ["train", "valid", "test"]:

    annotation_file = os.path.join(
        split,
        "_annotations.coco.json"
    )

    with open(annotation_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    categories = {
        c["id"]: c["name"]
        for c in data["categories"]
    }

    counts = Counter(
        ann["category_id"]
        for ann in data["annotations"]
    )

    print(f"\n========== {split.upper()} ==========")

    for category_id, class_name in categories.items():
        count = counts.get(category_id, 0)

        print(
            f"{category_id}: "
            f"{class_name:20} -> "
            f"{count}"
        )

    # Check invalid bounding boxes
    invalid_boxes = 0

    for ann in data["annotations"]:
        x, y, w, h = ann["bbox"]

        if w <= 0 or h <= 0:
            invalid_boxes += 1

    print("\nInvalid bounding boxes:", invalid_boxes)