import json
import os
import glob

# ===============================
# CONFIG
# ===============================
LABEL_NAME = "melon"
CLASS_ID = 0

JSON_DIR = "melon_dataset/labels/train"
IMG_DIR = "melon_dataset/images/train"
OUT_DIR = "melon_dataset/labels/train_yolo"

os.makedirs(OUT_DIR, exist_ok=True)

# ===============================
# CONVERT
# ===============================
for json_path in glob.glob(f"{JSON_DIR}/*.json"):
    with open(json_path, "r") as f:
        data = json.load(f)

    img_w = data["imageWidth"]
    img_h = data["imageHeight"]

    yolo_lines = []

    for shape in data["shapes"]:
        if shape["label"] != LABEL_NAME:
            continue

        points = shape["points"]
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]

        x_min = min(x_coords)
        x_max = max(x_coords)
        y_min = min(y_coords)
        y_max = max(y_coords)

        x_center = ((x_min + x_max) / 2) / img_w
        y_center = ((y_min + y_max) / 2) / img_h
        width = (x_max - x_min) / img_w
        height = (y_max - y_min) / img_h

        yolo_lines.append(
            f"{CLASS_ID} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        )

    txt_name = os.path.basename(json_path).replace(".json", ".txt")
    with open(os.path.join(OUT_DIR, txt_name), "w") as f:
        f.write("\n".join(yolo_lines))

print("✅ Konversi JSON → YOLO selesai")
