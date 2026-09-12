import os
import json
from collections import Counter, defaultdict

splits = ["train", "valid", "test"]

for split in splits:
    print(f"\n{'='*60}")
    print(f"{split.upper()} DATASET STATS")
    print(f"{'='*60}")

    ann_file = os.path.join(split, "_annotations.coco.json")

    with open(ann_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    images = data["images"]
    annotations = data["annotations"]

    # Image dimensions
    widths = Counter(img["width"] for img in images)
    heights = Counter(img["height"] for img in images)

    print("\nImage dimensions:")
    print("Most common widths :", widths.most_common(5))
    print("Most common heights:", heights.most_common(5))

    # Bounding box statistics
    class_boxes = defaultdict(list)

    for ann in annotations:
        x, y, w, h = ann["bbox"]

        area = w * h
        img_area = next(
            img["width"] * img["height"]
            for img in images
            if img["id"] == ann["image_id"]
        )

        relative_area = area / img_area

        class_boxes[ann["category_id"]].append(
            (w, h, relative_area)
        )

    print("\nBounding box statistics:")

    categories = {
        c["id"]: c["name"]
        for c in data["categories"]
    }

    for cid, boxes in sorted(class_boxes.items()):
        areas = [b[2] for b in boxes]

        print(
            f"{categories[cid]:20s} "
            f"count={len(boxes):5d} "
            f"avg_area={sum(areas)/len(areas):.4f} "
            f"small(<1%)={sum(a < 0.01 for a in areas):5d} "
            f"large(>25%)={sum(a > 0.25 for a in areas):5d}"
        )

    # Images with no annotations
    annotated_images = set(a["image_id"] for a in annotations)

    empty_images = [
        img["id"]
        for img in images
        if img["id"] not in annotated_images
    ]

    print("\nImages with ZERO annotations:", len(empty_images))