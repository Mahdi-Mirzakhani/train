import os
import yaml
import glob

def is_label_valid(line, num_classes):
    try:
        parts = list(map(float, line.strip().split()))
        if len(parts) != 5:
            return False
        class_id, x, y, w, h = parts
        if not (0 <= class_id < num_classes):
            return False
        if not all(0 <= v <= 1 for v in (x, y, w, h)):
            return False
        return True
    except:
        return False

def clean_labels(yaml_path):
    print(f"🔍 Reading config from: {yaml_path}")
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    nc = data.get('nc')
    if not nc:
        print("❌ 'nc' (number of classes) not found in YAML.")
        return

    datasets = ['train', 'val', 'test']
    for split in datasets:
        split_path = data.get(split)
        if not split_path:
            continue
        labels_dir = os.path.join(split_path, 'labels')
        images_dir = os.path.join(split_path, 'images')
        if not os.path.exists(labels_dir):
            print(f"⚠️ Labels path not found: {labels_dir}")
            continue

        print(f"\n📂 Checking labels in: {labels_dir}")
        txt_files = glob.glob(os.path.join(labels_dir, '*.txt'))
        removed_count = 0
        for txt_file in txt_files:
            with open(txt_file, 'r') as f:
                lines = f.readlines()

            if len(lines) == 0 or not all(is_label_valid(line, nc) for line in lines):
                # Delete label and image
                os.remove(txt_file)
                # Try to delete image with same name
                base = os.path.splitext(os.path.basename(txt_file))[0]
                for ext in ['.jpg', '.jpeg', '.png']:
                    img_path = os.path.join(images_dir, base + ext)
                    if os.path.exists(img_path):
                        os.remove(img_path)
                        break
                removed_count += 1
        print(f"✅ Removed {removed_count} invalid labels from '{split}' split.")

# ======================
# اجرا:
# فقط مسیر فایل YAML را اینجا بده
yaml_file_path = r"D:\\python\\dataset\\data.yaml"
clean_labels(yaml_file_path)
