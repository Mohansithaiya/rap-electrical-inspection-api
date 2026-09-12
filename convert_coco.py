import os
import json
import shutil

SOURCE_SPLITS = ["train", "valid", "test"]

# COCO category IDs we actually want
CATEGORY_MAP = {
    1: 0,  # arc
    2: 1,  # disconnector
    3: 2,  # disconnector_open
    4: 3,  # insulator
    5: 4,  # spark
    6: 5,  # switch
    7: 6,  # switchgear
    8: 7,  # transformer
}

CLASS_NAMES = [
    "arc",
    "disconnector",
    "disconnector_open",
    "insulator",
    "spark",
    "switch",
    "switchgear",
    "transformer",
]

OUTPUT_ROOT = "yolo_dataset"


def convert_split(split):

    source_dir = split
    annotation_file = os.path.join(
        source_dir,
        "_annotations.coco.json"
    )

    output_image_dir = os.path.join(
        OUTPUT_ROOT,
        "images",
        split
    )

    output_label_dir = os.path.join(
        OUTPUT_ROOT,
        "labels",
        split
    )

    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    with open(annotation_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    images = {
        img["id"]: img
        for img in data["images"]
    }

    annotations_by_image = {}

    for ann in data["annotations"]:

        category_id = ann["category_id"]

        # Ignore empty helmet-insulator class
        if category_id not in CATEGORY_MAP:
            continue

        annotations_by_image.setdefault(
            ann["image_id"], []
        ).append(ann)

    converted = 0
    skipped = 0

    for image_id, image in images.items():

        filename = image["file_name"]

        source_image = os.path.join(
            source_dir,
            filename
        )

        if not os.path.exists(source_image):
            print("WARNING - missing:", source_image)
            continue

        # Copy image
        destination_image = os.path.join(
            output_image_dir,
            filename
        )

        shutil.copy2(
            source_image,
            destination_image
        )

        label_file = os.path.splitext(filename)[0] + ".txt"

        label_path = os.path.join(
            output_label_dir,
            label_file
        )

        anns = annotations_by_image.get(
            image_id,
            []
        )

        with open(label_path, "w") as f:

            for ann in anns:

                x, y, w, h = ann["bbox"]

                img_w = image["width"]
                img_h = image["height"]

                # Convert COCO xywh → YOLO cxcywh
                x_center = (x + w / 2) / img_w
                y_center = (y + h / 2) / img_h

                width = w / img_w
                height = h / img_h

                class_id = CATEGORY_MAP[
                    ann["category_id"]
                ]

                f.write(
                    f"{class_id} "
                    f"{x_center:.6f} "
                    f"{y_center:.6f} "
                    f"{width:.6f} "
                    f"{height:.6f}\n"
                )

        if anns:
            converted += 1
        else:
            skipped += 1

    print(f"\n===== {split.upper()} =====")
    print("Images processed :", len(images))
    print("Images with labels:", converted)
    print("Empty images      :", skipped)


print("\nCOCO → YOLO CONVERSION")
print("======================")

os.makedirs(OUTPUT_ROOT, exist_ok=True)

for split in SOURCE_SPLITS:
    convert_split(split)

# Create data.yaml
yaml_path = os.path.join(
    OUTPUT_ROOT,
    "data.yaml"
)

with open(yaml_path, "w", encoding="utf-8") as f:

    f.write(
        f"""path: {OUTPUT_ROOT}
train: images/train
val: images/valid
test: images/test

nc: {len(CLASS_NAMES)}

names:
"""
    )

    for i, name in enumerate(CLASS_NAMES):
        f.write(f"  {i}: {name}\n")

print("\n================================")
print("CONVERSION COMPLETE")
print("================================")
print("Output:", os.path.abspath(OUTPUT_ROOT))
print("Classes:", CLASS_NAMES)
print("data.yaml created.")