import os
import shutil
from glob import glob
import cv2

# --------------------------
# فق\ط این دو مسیر را تغییر بده
INPUT_DIR = r"D:\\train\\new-dataset\\image3_dataset\\image3_dataset"   # مسیر دیتاست YOLOv8
OUTPUT_DIR = r"D:\\train\\new-dataset\\image3_dataset"  # مسیر خروجی WIDERFace
# --------------------------
os.makedirs(f"{OUTPUT_DIR}/images", exist_ok=True)
label_file = open(f"{OUTPUT_DIR}/label.txt", "w")

for file in os.listdir(INPUT_DIR):
    if not file.lower().endswith(".jpg"):
        continue

    img_path = os.path.join(INPUT_DIR, file)
    txt_path = img_path.replace(".jpg", ".txt")

    # کپی عکس
    shutil.copy(img_path, f"{OUTPUT_DIR}/images/{file}")

    # خواندن اندازه عکس
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    # نوشتن اسم عکس
    label_file.write(file + "\n")

    # اگر لیبل نبود
    if not os.path.exists(txt_path):
        label_file.write("0\n")
        continue

    with open(txt_path, "r") as f:
        lines = f.readlines()

    # تعداد چهره‌ها
    label_file.write(str(len(lines)) + "\n")

    for line in lines:
        cls, xc, yc, bw, bh = map(float, line.strip().split())

        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)

        # attribute صفر
        label_file.write(f"{x1} {y1} {x2} {y2} 0 0 0 0 0 0\n")

label_file.close()

print("DONE: Your single-folder dataset converted to WIDERFace!")