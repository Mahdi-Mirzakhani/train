import os
import shutil
import random
from pathlib import Path
import argparse

def create_yolo_structure(output_dir):
    """ساختار پوشه‌های YOLOv8 را ایجاد می‌کند"""
    dirs = [
        'train/images',
        'train/labels',
        'val/images', 
        'val/labels',
        'test/images',
        'test/labels'
    ]
    
    for dir_path in dirs:
        Path(output_dir, dir_path).mkdir(parents=True, exist_ok=True)
    
    print(f"ساختار YOLOv8 در {output_dir} ایجاد شد")

def get_image_label_pairs(input_dir):
    """جفت‌های عکس و لیبل را پیدا می‌کند"""
    input_path = Path(input_dir)
    
    # فرمت‌های پشتیبانی شده برای عکس
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    # فرمت‌های پشتیبانی شده برای لیبل
    label_extensions = {'.txt'}
    
    # پیدا کردن تمام فایل‌های عکس
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_path.glob(f'*{ext}'))
        image_files.extend(input_path.glob(f'*{ext.upper()}'))
    
    # پیدا کردن جفت‌های عکس و لیبل
    pairs = []
    for image_file in image_files:
        # نام فایل بدون پسوند
        base_name = image_file.stem
        
        # جستجو برای فایل لیبل متناظر
        label_file = None
        for ext in label_extensions:
            potential_label = input_path / f"{base_name}{ext}"
            if potential_label.exists():
                label_file = potential_label
                break
        
        if label_file:
            pairs.append((image_file, label_file))
        else:
            print(f"هشدار: فایل لیبل برای {image_file.name} پیدا نشد")
    
    return pairs

def split_dataset(pairs, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """دیتاست را تقسیم می‌کند"""
    # بررسی نسبت‌ها
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.001:
        raise ValueError("مجموع نسبت‌ها باید برابر 1 باشد")
    
    # مخلوط کردن داده‌ها
    random.shuffle(pairs)
    
    total = len(pairs)
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    
    train_pairs = pairs[:train_count]
    val_pairs = pairs[train_count:train_count + val_count]
    test_pairs = pairs[train_count + val_count:]
    
    return train_pairs, val_pairs, test_pairs

def copy_files(pairs, output_dir, split_name):
    """فایل‌ها را به مقصد کپی می‌کند"""
    images_dir = Path(output_dir) / split_name / 'images'
    labels_dir = Path(output_dir) / split_name / 'labels'
    
    for image_file, label_file in pairs:
        # کپی عکس
        shutil.copy2(image_file, images_dir / image_file.name)
        # کپی لیبل
        shutil.copy2(label_file, labels_dir / label_file.name)

def create_yaml_config(output_dir, class_names=None):
    """فایل پیکربندی YAML برای YOLOv8 ایجاد می‌کند"""
    yaml_content = f"""# YOLOv8 Dataset Configuration
path: {os.path.abspath(output_dir)}
train: train/images
val: val/images
test: test/images

# Classes
nc: {len(class_names) if class_names else 'NUM_CLASSES'}
names: {class_names if class_names else ['class0', 'class1', 'class2']}
"""
    
    with open(Path(output_dir) / 'data.yaml', 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    
    print(f"فایل پیکربندی data.yaml ایجاد شد")

def convert_to_yolo(input_dir, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, class_names=None):
    """تابع اصلی تبدیل"""
    print(f"شروع تبدیل دیتاست از {input_dir} به {output_dir}")
    
    # ایجاد ساختار YOLOv8
    create_yolo_structure(output_dir)
    
    # پیدا کردن جفت‌های عکس و لیبل
    pairs = get_image_label_pairs(input_dir)
    
    if not pairs:
        print("هیچ جفت عکس و لیبل معتبری پیدا نشد!")
        return
    
    print(f"تعداد کل جفت‌های عکس و لیبل: {len(pairs)}")
    
    # تقسیم دیتاست
    train_pairs, val_pairs, test_pairs = split_dataset(pairs, train_ratio, val_ratio, test_ratio)
    
    print(f"تقسیم‌بندی:")
    print(f"  آموزش: {len(train_pairs)} ({len(train_pairs)/len(pairs)*100:.1f}%)")
    print(f"  اعتبارسنجی: {len(val_pairs)} ({len(val_pairs)/len(pairs)*100:.1f}%)")
    print(f"  تست: {len(test_pairs)} ({len(test_pairs)/len(pairs)*100:.1f}%)")
    
    # کپی فایل‌ها
    copy_files(train_pairs, output_dir, 'train')
    copy_files(val_pairs, output_dir, 'val')
    copy_files(test_pairs, output_dir, 'test')
    
    # ایجاد فایل پیکربندی
    create_yaml_config(output_dir, class_names)
    
    print("تبدیل با موفقیت انجام شد!")

def main():
    parser = argparse.ArgumentParser(description='تبدیل دیتاست به ساختار YOLOv8')
    parser.add_argument('input_dir', help='پوشه ورودی حاوی عکس‌ها و لیبل‌ها')
    parser.add_argument('output_dir', help='پوشه خروجی با ساختار YOLOv8')
    parser.add_argument('--train-ratio', type=float, default=0.7, help='نسبت داده‌های آموزش (پیش‌فرض: 0.7)')
    parser.add_argument('--val-ratio', type=float, default=0.2, help='نسبت داده‌های اعتبارسنجی (پیش‌فرض: 0.2)')
    parser.add_argument('--test-ratio', type=float, default=0.1, help='نسبت داده‌های تست (پیش‌فرض: 0.1)')
    parser.add_argument('--classes', nargs='*', help='نام کلاس‌ها (اختیاری)')
    
    args = parser.parse_args()
    
    convert_to_yolo(
        args.input_dir,
        args.output_dir,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.classes
    )

if __name__ == '__main__':
    # آدرس پوشه ورودی (حاوی عکس‌ها و لیبل‌ها)
    input_folder = r"D:\\python\\dataset\\drone\\main_dataset"
    
    # آدرس پوشه خروجی (جایی که ساختار YOLOv8 ایجاد می‌شود)
    output_folder = r"D:\\python\\dataset\\drone\\main_dataset\\test"
    
    # تبدیل دیتاست
    convert_to_yolo(input_folder, output_folder)
    
    # اگر می‌خواهید از خط فرمان استفاده کنید، کامنت بالا را حذف و این خط را فعال کنید:
    # main()