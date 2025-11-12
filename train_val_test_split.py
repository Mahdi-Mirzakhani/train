import os
import shutil
import random
import json
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import threading


class DatasetSplitter:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Dataset Splitter - تقسیم‌کننده دیتاست")
        self.root.geometry("700x650")
        self.root.configure(bg='#f0f0f0')
        
        # متغیرها
        self.input_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.train_var = tk.StringVar(value="70")
        self.val_var = tk.StringVar(value="20")
        self.test_var = tk.StringVar(value="10")
        self.preserve_structure = tk.BooleanVar(value=False)
        self.copy_mode = tk.StringVar(value="copy")
        self.has_labels = tk.BooleanVar(value=False)
        
        self.total_files = 0
        self.processed_files = 0
        self.file_info = {}
        
        self.create_widgets()
        self.center_window()
    
    def center_window(self):
        """پنجره را وسط صفحه قرار می‌دهد"""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
    
    def create_widgets(self):
        # عنوان اصلی
        title_frame = tk.Frame(self.root, bg='#2c3e50', height=60)
        title_frame.pack(fill='x', padx=5, pady=5)
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(title_frame, text="🖼️ تقسیم‌کننده دیتاست تصاویر", 
                              font=('Arial', 16, 'bold'), fg='white', bg='#2c3e50')
        title_label.pack(expand=True)
        
        # فریم اصلی
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # بخش انتخاب مسیرها
        self.create_path_section(main_frame)
        
        # بخش تنظیمات
        self.create_settings_section(main_frame)
        
        # بخش پیش‌نمایش
        self.create_preview_section(main_frame)
        
        # بخش نوار پیشرفت و دکمه‌ها
        self.create_control_section(main_frame)
    
    def create_path_section(self, parent):
        # فریم مسیرها
        path_frame = tk.LabelFrame(parent, text="📁 مسیرها", font=('Arial', 10, 'bold'), 
                                  bg='#f0f0f0', fg='#2c3e50')
        path_frame.pack(fill='x', pady=5)
        
        # مسیر ورودی
        tk.Label(path_frame, text="مسیر دیتاست:", bg='#f0f0f0').grid(row=0, column=0, sticky='w', padx=5, pady=2)
        input_frame = tk.Frame(path_frame, bg='#f0f0f0')
        input_frame.grid(row=1, column=0, columnspan=2, sticky='ew', padx=5, pady=2)
        
        self.input_entry = tk.Entry(input_frame, textvariable=self.input_var, width=50)
        self.input_entry.pack(side='left', fill='x', expand=True)
        tk.Button(input_frame, text="انتخاب...", command=self.browse_input, 
                 bg='#3498db', fg='white').pack(side='right', padx=(5,0))
        
        # مسیر خروجی
        tk.Label(path_frame, text="مسیر خروجی:", bg='#f0f0f0').grid(row=2, column=0, sticky='w', padx=5, pady=2)
        output_frame = tk.Frame(path_frame, bg='#f0f0f0')
        output_frame.grid(row=3, column=0, columnspan=2, sticky='ew', padx=5, pady=2)
        
        self.output_entry = tk.Entry(output_frame, textvariable=self.output_var, width=50)
        self.output_entry.pack(side='left', fill='x', expand=True)
        tk.Button(output_frame, text="انتخاب...", command=self.browse_output,
                 bg='#3498db', fg='white').pack(side='right', padx=(5,0))
        
        path_frame.columnconfigure(0, weight=1)
    
    def create_settings_section(self, parent):
        # فریم تنظیمات
        settings_frame = tk.LabelFrame(parent, text="⚙️ تنظیمات", font=('Arial', 10, 'bold'),
                                     bg='#f0f0f0', fg='#2c3e50')
        settings_frame.pack(fill='x', pady=5)
        
        # درصدهای تقسیم‌بندی
        split_frame = tk.Frame(settings_frame, bg='#f0f0f0')
        split_frame.pack(fill='x', padx=5, pady=5)
        
        tk.Label(split_frame, text="درصد تقسیم‌بندی:", bg='#f0f0f0', font=('Arial', 9, 'bold')).pack(anchor='w')
        
        percent_frame = tk.Frame(split_frame, bg='#f0f0f0')
        percent_frame.pack(fill='x', pady=2)
        
        # Train
        tk.Label(percent_frame, text="Train:", bg='#f0f0f0').grid(row=0, column=0, padx=5)
        train_spin = tk.Spinbox(percent_frame, from_=0, to=100, textvariable=self.train_var, 
                               width=5, command=self.update_percentages)
        train_spin.grid(row=0, column=1, padx=2)
        tk.Label(percent_frame, text="%", bg='#f0f0f0').grid(row=0, column=2, padx=2)
        
        # Validation
        tk.Label(percent_frame, text="Val:", bg='#f0f0f0').grid(row=0, column=3, padx=5)
        val_spin = tk.Spinbox(percent_frame, from_=0, to=100, textvariable=self.val_var,
                             width=5, command=self.update_percentages)
        val_spin.grid(row=0, column=4, padx=2)
        tk.Label(percent_frame, text="%", bg='#f0f0f0').grid(row=0, column=5, padx=2)
        
        # Test
        tk.Label(percent_frame, text="Test:", bg='#f0f0f0').grid(row=0, column=6, padx=5)
        test_spin = tk.Spinbox(percent_frame, from_=0, to=100, textvariable=self.test_var,
                              width=5, command=self.update_percentages)
        test_spin.grid(row=0, column=7, padx=2)
        tk.Label(percent_frame, text="%", bg='#f0f0f0').grid(row=0, column=8, padx=2)
        
        # نمایش مجموع
        self.total_label = tk.Label(percent_frame, text="مجموع: 100%", bg='#f0f0f0', 
                                   fg='green', font=('Arial', 9, 'bold'))
        self.total_label.grid(row=0, column=9, padx=10)
        
        # گزینه‌های اضافی
        options_frame = tk.Frame(settings_frame, bg='#f0f0f0')
        options_frame.pack(fill='x', padx=5, pady=5)
        
        tk.Checkbutton(options_frame, text="حفظ ساختار پوشه‌ای (برای دیتاست کلاس‌بندی شده)", 
                      variable=self.preserve_structure, bg='#f0f0f0').pack(anchor='w')
        
        tk.Checkbutton(options_frame, text="دیتاست دارای فایل‌های Label است (YOLO, Object Detection)", 
                      variable=self.has_labels, bg='#f0f0f0', fg='#e74c3c',
                      font=('Arial', 9, 'bold')).pack(anchor='w')
        
        mode_frame = tk.Frame(options_frame, bg='#f0f0f0')
        mode_frame.pack(fill='x', pady=2)
        tk.Label(mode_frame, text="حالت عملیات:", bg='#f0f0f0').pack(side='left')
        tk.Radiobutton(mode_frame, text="کپی فایل‌ها", variable=self.copy_mode, value="copy", bg='#f0f0f0').pack(side='left', padx=10)
        tk.Radiobutton(mode_frame, text="انتقال فایل‌ها", variable=self.copy_mode, value="move", bg='#f0f0f0').pack(side='left')
    
    def create_preview_section(self, parent):
        # فریم پیش‌نمایش
        preview_frame = tk.LabelFrame(parent, text="👁️ پیش‌نمایش", font=('Arial', 10, 'bold'),
                                    bg='#f0f0f0', fg='#2c3e50')
        preview_frame.pack(fill='both', expand=True, pady=5)
        
        # دکمه اسکن
        scan_frame = tk.Frame(preview_frame, bg='#f0f0f0')
        scan_frame.pack(fill='x', padx=5, pady=5)
        
        tk.Button(scan_frame, text="🔍 اسکن دیتاست", command=self.scan_dataset,
                 bg='#e74c3c', fg='white', font=('Arial', 9, 'bold')).pack(side='left')
        
        self.info_label = tk.Label(scan_frame, text="", bg='#f0f0f0', fg='#7f8c8d')
        self.info_label.pack(side='left', padx=10)
        
        # جدول اطلاعات
        self.tree = ttk.Treeview(preview_frame, columns=('type', 'count', 'train', 'val', 'test'), 
                                show='tree headings', height=6)
        
        self.tree.heading('#0', text='نوع/کلاس')
        self.tree.heading('type', text='نوع فایل')
        self.tree.heading('count', text='تعداد کل')
        self.tree.heading('train', text='Train')
        self.tree.heading('val', text='Validation')
        self.tree.heading('test', text='Test')
        
        # تنظیم عرض ستون‌ها
        self.tree.column('#0', width=150)
        self.tree.column('type', width=80)
        self.tree.column('count', width=80)
        self.tree.column('train', width=80)
        self.tree.column('val', width=80)
        self.tree.column('test', width=80)
        
        # اسکرول بار
        scrollbar = ttk.Scrollbar(preview_frame, orient='vertical', command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        self.tree.pack(side='left', fill='both', expand=True, padx=5, pady=5)
        scrollbar.pack(side='right', fill='y')
    
    def create_control_section(self, parent):
        # فریم کنترل
        control_frame = tk.Frame(parent, bg='#f0f0f0')
        control_frame.pack(fill='x', pady=5)
        
        # نوار پیشرفت
        self.progress_var = tk.DoubleVar()
        self.progress = ttk.Progressbar(control_frame, variable=self.progress_var, 
                                       maximum=100, length=400)
        self.progress.pack(pady=5)
        
        self.progress_label = tk.Label(control_frame, text="", bg='#f0f0f0', fg='#7f8c8d')
        self.progress_label.pack()
        
        # دکمه‌ها
        button_frame = tk.Frame(control_frame, bg='#f0f0f0')
        button_frame.pack(pady=10)
        
        self.start_button = tk.Button(button_frame, text="▶️ شروع تقسیم‌بندی", 
                                     command=self.start_process, bg='#27ae60', fg='white',
                                     font=('Arial', 11, 'bold'), padx=20)
        self.start_button.pack(side='left', padx=5)
        
        tk.Button(button_frame, text="📋 ذخیره گزارش", command=self.save_report,
                 bg='#f39c12', fg='white', padx=15).pack(side='left', padx=5)
        
        tk.Button(button_frame, text="❌ خروج", command=self.root.quit,
                 bg='#e74c3c', fg='white', padx=15).pack(side='left', padx=5)
    
    def update_percentages(self):
        """به‌روزرسانی مجموع درصدها"""
        try:
            total = int(self.train_var.get()) + int(self.val_var.get()) + int(self.test_var.get())
            color = 'green' if total == 100 else 'red'
            self.total_label.config(text=f"مجموع: {total}%", fg=color)
            self.update_preview()
        except ValueError:
            self.total_label.config(text="مجموع: ???%", fg='red')
    
    def browse_input(self):
        folder = filedialog.askdirectory(title="انتخاب پوشه دیتاست")
        if folder:
            self.input_var.set(folder)
    
    def browse_output(self):
        folder = filedialog.askdirectory(title="انتخاب پوشه خروجی")
        if folder:
            self.output_var.set(folder)
    
    def get_image_files(self, directory):
        """جمع‌آوری فایل‌های تصویری"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
        image_files = []
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                if os.path.splitext(file.lower())[1] in image_extensions:
                    image_files.append(os.path.join(root, file))
        
        return image_files
    
    def find_label_file(self, image_path, dataset_dir):
        """پیدا کردن فایل label متناظر با تصویر"""
        # پسوندهای معمول label
        label_extensions = ['.txt', '.xml', '.json']
        
        # نام فایل بدون پسوند
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # مسیر نسبی تصویر
        rel_path = os.path.relpath(os.path.dirname(image_path), dataset_dir)
        
        # جستجو برای فایل label
        possible_label_dirs = [
            os.path.dirname(image_path),  # همان پوشه تصویر
            os.path.join(dataset_dir, 'labels'),  # پوشه labels در ریشه
            os.path.join(dataset_dir, rel_path.replace('images', 'labels')),  # labels موازی با images
        ]
        
        for label_dir in possible_label_dirs:
            if os.path.exists(label_dir):
                for ext in label_extensions:
                    label_path = os.path.join(label_dir, image_name + ext)
                    if os.path.exists(label_path):
                        return label_path
        
        return None
    
    def scan_dataset(self):
        """اسکن دیتاست و نمایش اطلاعات"""
        if not self.input_var.get():
            messagebox.showerror("خطا", "لطفاً مسیر دیتاست را انتخاب کنید")
            return
        
        if not os.path.exists(self.input_var.get()):
            messagebox.showerror("خطا", "مسیر دیتاست معتبر نیست")
            return
        
        # پاک کردن جدول قبلی
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        dataset_dir = self.input_var.get()
        self.info_label.config(text="در حال اسکن...")
        self.root.update()
        
        # جمع‌آوری اطلاعات
        all_images = self.get_image_files(dataset_dir)
        self.total_files = len(all_images)
        
        if self.total_files == 0:
            self.info_label.config(text="هیچ تصویری یافت نشد!")
            return
        
        # شمارش labelها اگر فعال باشد
        labels_count = 0
        if self.has_labels.get():
            for img in all_images:
                if self.find_label_file(img, dataset_dir):
                    labels_count += 1
        
        # تحلیل ساختار
        if self.preserve_structure.get():
            # تحلیل بر اساس کلاس‌ها
            class_info = {}
            for img_path in all_images:
                rel_path = os.path.relpath(img_path, dataset_dir)
                class_name = os.path.dirname(rel_path)
                if not class_name:
                    class_name = "root"
                
                if class_name not in class_info:
                    class_info[class_name] = []
                class_info[class_name].append(img_path)
            
            self.file_info = class_info
            
            # اضافه کردن به جدول
            for class_name, files in class_info.items():
                count = len(files)
                train_count = int(count * int(self.train_var.get()) / 100)
                val_count = int(count * int(self.val_var.get()) / 100)
                test_count = count - train_count - val_count
                
                self.tree.insert('', 'end', text=class_name, values=(
                    'کلاس', count, train_count, val_count, test_count
                ))
        else:
            # تحلیل کلی
            file_types = {}
            for img_path in all_images:
                ext = os.path.splitext(img_path.lower())[1]
                if ext not in file_types:
                    file_types[ext] = 0
                file_types[ext] += 1
            
            # اضافه کردن آمار کلی
            train_count = int(self.total_files * int(self.train_var.get()) / 100)
            val_count = int(self.total_files * int(self.val_var.get()) / 100)
            test_count = self.total_files - train_count - val_count
            
            self.tree.insert('', 'end', text='کل فایل‌ها', values=(
                'همه', self.total_files, train_count, val_count, test_count
            ))
            
            # اضافه کردن اطلاعات نوع فایل
            for ext, count in file_types.items():
                train_count = int(count * int(self.train_var.get()) / 100)
                val_count = int(count * int(self.val_var.get()) / 100)
                test_count = count - train_count - val_count
                
                self.tree.insert('', 'end', text=f'  {ext}', values=(
                    'فرمت', count, train_count, val_count, test_count
                ))
        
        info_text = f"تعداد کل: {self.total_files} تصویر"
        if self.has_labels.get():
            info_text += f" | Labels: {labels_count}"
        self.info_label.config(text=info_text)
    
    def update_preview(self):
        """به‌روزرسانی پیش‌نمایش بعد از تغییر درصدها"""
        if hasattr(self, 'file_info') and self.file_info:
            self.scan_dataset()
    
    def split_dataset_with_structure(self, dataset_dir, output_dir, train_split, val_split, test_split):
        """تقسیم دیتاست با حفظ ساختار"""
        splits = ['train', 'val', 'test']
        
        # ساخت ساختار پوشه‌ها
        for split in splits:
            for class_name in self.file_info.keys():
                # ساخت پوشه images
                images_dir = os.path.join(output_dir, split, 'images', class_name)
                os.makedirs(images_dir, exist_ok=True)
                
                # ساخت پوشه labels اگر نیاز باشد
                if self.has_labels.get():
                    labels_dir = os.path.join(output_dir, split, 'labels', class_name)
                    os.makedirs(labels_dir, exist_ok=True)
        
        total_processed = 0
        processed_files_set = set()
        
        for class_name, files in self.file_info.items():
            files_copy = files.copy()
            random.shuffle(files_copy)
            
            total = len(files_copy)
            train_count = int(total * train_split / 100)
            val_count = int(total * val_split / 100)
            
            train_files = files_copy[:train_count]
            val_files = files_copy[train_count:train_count + val_count]
            test_files = files_copy[train_count + val_count:]
            
            split_files = {
                'train': train_files,
                'val': val_files,
                'test': test_files
            }
            
            for split, file_list in split_files.items():
                for file_path in file_list:
                    if file_path in processed_files_set:
                        print(f"⚠️ هشدار: فایل تکراری: {file_path}")
                        continue
                    
                    filename = os.path.basename(file_path)
                    
                    # کپی تصویر
                    dest_image_path = os.path.join(output_dir, split, 'images', class_name, filename)
                    
                    if os.path.exists(dest_image_path):
                        base, ext = os.path.splitext(filename)
                        counter = 1
                        while os.path.exists(dest_image_path):
                            new_filename = f"{base}_{counter}{ext}"
                            dest_image_path = os.path.join(output_dir, split, 'images', class_name, new_filename)
                            counter += 1
                        filename = os.path.basename(dest_image_path)
                    
                    try:
                        if self.copy_mode.get() == "copy":
                            shutil.copy2(file_path, dest_image_path)
                        else:
                            shutil.move(file_path, dest_image_path)
                        
                        # کپی label اگر وجود داشته باشد
                        if self.has_labels.get():
                            label_path = self.find_label_file(file_path, dataset_dir)
                            if label_path:
                                label_filename = os.path.splitext(filename)[0] + os.path.splitext(label_path)[1]
                                dest_label_path = os.path.join(output_dir, split, 'labels', class_name, label_filename)
                                
                                if self.copy_mode.get() == "copy":
                                    shutil.copy2(label_path, dest_label_path)
                                else:
                                    shutil.move(label_path, dest_label_path)
                        
                        processed_files_set.add(file_path)
                        total_processed += 1
                        progress = (total_processed / self.total_files) * 100
                        self.progress_var.set(progress)
                        self.progress_label.config(text=f"پردازش: {total_processed}/{self.total_files} ({split}/{class_name})")
                        self.root.update()
                    except Exception as e:
                        print(f"خطا در پردازش {file_path}: {str(e)}")
    
    def split_dataset_simple(self, dataset_dir, output_dir, train_split, val_split, test_split):
        """تقسیم ساده دیتاست"""
        splits = ['train', 'val', 'test']
        
        # ساخت ساختار پوشه‌ها
        for split in splits:
            os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
            if self.has_labels.get():
                os.makedirs(os.path.join(output_dir, split, 'labels'), exist_ok=True)
        
        all_images = self.get_image_files(dataset_dir)
        random.shuffle(all_images)
        
        total = len(all_images)
        train_count = int(total * train_split / 100)
        val_count = int(total * val_split / 100)
        
        train_files = all_images[:train_count]
        val_files = all_images[train_count:train_count + val_count]
        test_files = all_images[train_count + val_count:]
        
        splits_data = {
            'train': train_files,
            'val': val_files,
            'test': test_files
        }
        
        total_processed = 0
        processed_files_set = set()
        
        for split, files in splits_data.items():
            for file_path in files:
                if file_path in processed_files_set:
                    print(f"⚠️ هشدار: فایل تکراری: {file_path}")
                    continue
                
                filename = os.path.basename(file_path)
                dest_image_path = os.path.join(output_dir, split, 'images', filename)
                
                if os.path.exists(dest_image_path):
                    base, ext = os.path.splitext(filename)
                    counter = 1
                    while os.path.exists(dest_image_path):
                        new_filename = f"{base}_{counter}{ext}"
                        dest_image_path = os.path.join(output_dir, split, 'images', new_filename)
                        counter += 1
                    filename = os.path.basename(dest_image_path)
                
                try:
                    if self.copy_mode.get() == "copy":
                        shutil.copy2(file_path, dest_image_path)
                    else:
                        shutil.move(file_path, dest_image_path)
                    
                    # کپی label اگر وجود داشته باشد
                    if self.has_labels.get():
                        label_path = self.find_label_file(file_path, dataset_dir)
                        if label_path:
                            label_filename = os.path.splitext(filename)[0] + os.path.splitext(label_path)[1]
                            dest_label_path = os.path.join(output_dir, split, 'labels', label_filename)
                            
                            if self.copy_mode.get() == "copy":
                                shutil.copy2(label_path, dest_label_path)
                            else:
                                shutil.move(label_path, dest_label_path)
                    
                    processed_files_set.add(file_path)
                    total_processed += 1
                    progress = (total_processed / total) * 100
                    self.progress_var.set(progress)
                    self.progress_label.config(text=f"پردازش: {total_processed}/{total} ({split})")
                    self.root.update()
                except Exception as e:
                    print(f"خطا در پردازش {file_path}: {str(e)}")
    
    def process_dataset(self):
        """پردازش اصلی دیتاست"""
        try:
            dataset_dir = self.input_var.get()
            output_dir = self.output_var.get()
            train_split = int(self.train_var.get())
            val_split = int(self.val_var.get())
            test_split = int(self.test_var.get())
            
            self.progress_var.set(0)
            self.start_button.config(state='disabled', text="در حال پردازش...")
            
            if self.preserve_structure.get() and hasattr(self, 'file_info'):
                self.split_dataset_with_structure(dataset_dir, output_dir, train_split, val_split, test_split)
            else:
                self.split_dataset_simple(dataset_dir, output_dir, train_split, val_split, test_split)
            
            self.progress_var.set(100)
            self.progress_label.config(text="✅ تقسیم‌بندی با موفقیت انجام شد!")
            
            msg = "تقسیم‌بندی تصاویر با موفقیت انجام شد! ✅\n\n"
            msg += "ساختار خروجی:\n"
            msg += "├── train/\n"
            msg += "│   ├── images/\n"
            if self.has_labels.get():
                msg += "│   └── labels/\n"
            msg += "├── val/\n"
            msg += "│   ├── images/\n"
            if self.has_labels.get():
                msg += "│   └── labels/\n"
            msg += "└── test/\n"
            msg += "    ├── images/\n"
            if self.has_labels.get():
                msg += "    └── labels/"
            
            messagebox.showinfo("موفقیت", msg)
            
        except Exception as e:
            messagebox.showerror("خطا", f"خطا در پردازش: {str(e)}")
        finally:
            self.start_button.config(state='normal', text="▶️ شروع تقسیم‌بندی")
    
    def start_process(self):
        """شروع فرآیند تقسیم‌بندی"""
        # بررسی‌های اولیه
        if not self.input_var.get():
            messagebox.showerror("خطا", "لطفاً مسیر دیتاست را انتخاب کنید")
            return
        
        if not self.output_var.get():
            messagebox.showerror("خطا", "لطفاً مسیر خروجی را انتخاب کنید")
            return
        
        try:
            train_split = int(self.train_var.get())
            val_split = int(self.val_var.get())
            test_split = int(self.test_var.get())
        except ValueError:
            messagebox.showerror("خطا", "لطفاً درصدهای معتبر وارد کنید")
            return
        
        if train_split + val_split + test_split != 100:
            messagebox.showerror("خطا", "مجموع درصدها باید 100 باشد")
            return
        
        if not os.path.exists(self.input_var.get()):
            messagebox.showerror("خطا", "مسیر دیتاست معتبر نیست")
            return
        
        # ایجاد پوشه خروجی
        if not os.path.exists(self.output_var.get()):
            try:
                os.makedirs(self.output_var.get())
            except:
                messagebox.showerror("خطا", "نمی‌توان پوشه خروجی را ایجاد کرد")
                return
        
        # اجرا در thread جداگانه
        thread = threading.Thread(target=self.process_dataset)
        thread.daemon = True
        thread.start()
    
    def save_report(self):
        """ذخیره گزارش تقسیم‌بندی"""
        if not hasattr(self, 'file_info') or not self.file_info:
            messagebox.showwarning("هشدار", "ابتدا دیتاست را اسکن کنید")
            return
        
        filename = filedialog.asksaveasfilename(
            title="ذخیره گزارش",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("Text files", "*.txt")]
        )
        
        if filename:
            try:
                report = {
                    'timestamp': datetime.now().isoformat(),
                    'input_dir': self.input_var.get(),
                    'output_dir': self.output_var.get(),
                    'splits': {
                        'train': int(self.train_var.get()),
                        'val': int(self.val_var.get()),
                        'test': int(self.test_var.get())
                    },
                    'preserve_structure': self.preserve_structure.get(),
                    'has_labels': self.has_labels.get(),
                    'copy_mode': self.copy_mode.get(),
                    'total_files': self.total_files,
                    'file_info': {k: len(v) if isinstance(v, list) else v for k, v in self.file_info.items()}
                }
                
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(report, f, ensure_ascii=False, indent=2)
                
                messagebox.showinfo("موفقیت", "گزارش با موفقیت ذخیره شد")
            except Exception as e:
                messagebox.showerror("خطا", f"خطا در ذخیره گزارش: {str(e)}")
    
    def run(self):
        """اجرای برنامه"""
        self.root.mainloop()


if __name__ == "__main__":
    app = DatasetSplitter()
    app.run()