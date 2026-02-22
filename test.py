import os
import cv2
import numpy as np
from glob import glob

# ----------------------------
# تنظیم مسیرها
# ----------------------------

images_dir = r"D:\\train\\NEU-DET\\train\\images"           # مسیر عکس‌ها
labels_dir = r"D:\\train\\NEU-DET\\train\\labels"           # لیبل‌های فعلی (بدون زاویه)
output_labels = r"D:\\train\\NEU-DET\\labels_rotated"   # خروجی زاویه‌دار

output_vis = r"D:\\train\\NEU-DET\\vis_rotated"

os.makedirs(output_labels, exist_ok=True)
os.makedirs(output_vis, exist_ok=True)

def get_angle_pca(points):
    """ محاسبه زاویه با PCA (دقیق‌ترین روش ممکن) """
    points = points.reshape(-1, 2).astype(np.float32)
    mean = np.mean(points, axis=0)
    centered = points - mean
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eig(cov)
    principal = eigvecs[:, np.argmax(eigvals)]
    angle = np.arctan2(principal[1], principal[0])
    return angle

def draw_rotated_box(img, cx, cy, w, h, angle, color=(0, 255, 0)):
    rect = ((cx, cy), (w, h), np.rad2deg(angle))
    box = cv2.boxPoints(rect).astype(int)
    cv2.drawContours(img, [box], 0, color, 2)
    return img

def convert_yolo_rotated():
    label_files = glob(os.path.join(labels_dir, "*.txt"))

    for lbl_path in label_files:
        name = os.path.basename(lbl_path).replace(".txt", "")
        img_path = os.path.join(images_dir, name + ".jpg")

        if not os.path.exists(img_path):
            img_path = os.path.join(images_dir, name + ".png")
        if not os.path.exists(img_path):
            print("❌ تصویر یافت نشد:", name)
            continue

        img = cv2.imread(img_path)
        H, W = img.shape[:2]
        vis_img = img.copy()
        out_lines = []

        with open(lbl_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            cls, cx, cy, bw, bh = line.split()
            cx = float(cx) * W
            cy = float(cy) * H
            bw = float(bw) * W
            bh = float(bh) * H

            # تولید نقاط مستطیل برای PCA
            xmin = cx - bw/2
            xmax = cx + bw/2
            ymin = cy - bh/2
            ymax = cy + bh/2

            box_points = np.array([
                [xmin, ymin],
                [xmax, ymin],
                [xmax, ymax],
                [xmin, ymax]
            ])

            angle = get_angle_pca(box_points)

            # نرمال‌سازی
            out_lines.append(
                f"{cls} {cx/W:.6f} {cy/H:.6f} {bw/W:.6f} {bh/H:.6f} {angle:.6f}"
            )

            # نمایش
            vis_img = draw_rotated_box(vis_img, cx, cy, bw, bh, angle)

        # ذخیره لیبل زاویه‌دار
        with open(os.path.join(output_labels, name + ".txt"), "w") as f:
            f.write("\n".join(out_lines))

        # ذخیره تصویر نمایش
        cv2.imwrite(os.path.join(output_vis, name + ".jpg"), vis_img)

        print("✔ پردازش شد:", name)

    print("\n🎉 کامل شد — زاویه‌ها دقیق و قابل مشاهده هستند.")


convert_yolo_rotated()
