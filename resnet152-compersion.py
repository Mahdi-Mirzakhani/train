import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
from PIL import Image, ImageTk, ImageDraw
import numpy as np
from pathlib import Path
import threading
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torch.nn import functional as F
from sklearn.metrics.pairwise import cosine_similarity
import time
import subprocess
import platform

class AdvancedImageComparator:
    def __init__(self, root):
        self.root = root
        self.root.title("مقایسه‌کننده پیشرفته تصاویر با GPU")
        self.root.geometry("1600x900")
        self.root.configure(bg='#1a1a2e')
        
        self.folder_paths = []
        self.duplicates = []
        self.device = None
        self.model = None
        self.current_group = None
        self.photo_references = []
        
        self.setup_ui()
        self.initialize_model()
    
    def initialize_model(self):
        """Initialize advanced deep learning model with perceptual features"""
        try:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                gpu_name = torch.cuda.get_device_name(0)
                self.gpu_label.config(text=f"✅ GPU فعال: {gpu_name}")
            else:
                self.device = torch.device('cpu')
                self.gpu_label.config(text="⚠️ GPU یافت نشد - استفاده از CPU")
            
            # استفاده از EfficientNet-B7 که دقیق‌تر از ResNet است
            from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
            
            self.model = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
            # حذف لایه classification برای استخراج features
            self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
            self.model.to(self.device)
            self.model.eval()
            
            self.gpu_label.config(text=f"✅ مدل پیشرفته: EfficientNet-B4 | GPU: {gpu_name if torch.cuda.is_available() else 'CPU'}")
            
        except Exception as e:
            # اگر EfficientNet بارگذاری نشد، از ResNet152 استفاده کن
            try:
                self.model = models.resnet152(pretrained=True)
                self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
                self.model.to(self.device)
                self.model.eval()
                self.gpu_label.config(text=f"✅ مدل: ResNet152")
            except:
                messagebox.showerror("خطا", f"خطا در بارگذاری مدل: {str(e)}")
    
    def setup_ui(self):
        main_container = tk.Frame(self.root, bg='#1a1a2e')
        main_container.pack(fill='both', expand=True)
        
        left_panel = tk.Frame(main_container, bg='#1a1a2e', width=500)
        left_panel.pack(side='left', fill='both', padx=10, pady=10)
        left_panel.pack_propagate(False)
        
        right_panel = tk.Frame(main_container, bg='#16213e', relief='solid', bd=2)
        right_panel.pack(side='right', fill='both', expand=True, padx=10, pady=10)
        
        self.setup_left_panel(left_panel)
        self.setup_right_panel(right_panel)
    
    def setup_left_panel(self, parent):
        header = tk.Frame(parent, bg='#16213e', height=100)
        header.pack(fill='x', pady=(0, 10))
        
        title = tk.Label(header, text="🚀 مقایسه‌کننده تصاویر", 
                        font=('Arial', 18, 'bold'), bg='#16213e', fg='#00d4ff')
        title.pack(pady=8)
        
        subtitle = tk.Label(header, text="EfficientNet-B4 + Multi-Scale Analysis", 
                           font=('Arial', 10), bg='#16213e', fg='#a0a0a0')
        subtitle.pack()
        
        gpu_frame = tk.Frame(parent, bg='#0f3460', relief='solid', bd=2)
        gpu_frame.pack(fill='x', pady=5)
        
        self.gpu_label = tk.Label(gpu_frame, text="⏳ در حال بررسی GPU...", 
                                 bg='#0f3460', fg='#00ff88', 
                                 font=('Arial', 9, 'bold'), pady=5)
        self.gpu_label.pack()
        
        folder_section = tk.LabelFrame(parent, text="📁 پوشه‌های انتخاب شده", 
                                       bg='#16213e', fg='#00d4ff',
                                       font=('Arial', 10, 'bold'), 
                                       padx=10, pady=10, relief='solid', bd=2)
        folder_section.pack(fill='both', expand=False, pady=8)
        
        folder_list_frame = tk.Frame(folder_section, bg='#16213e')
        folder_list_frame.pack(fill='both', expand=True)
        
        folder_scrollbar = tk.Scrollbar(folder_list_frame)
        folder_scrollbar.pack(side='right', fill='y')
        
        self.folder_listbox = tk.Listbox(folder_list_frame, 
                                         yscrollcommand=folder_scrollbar.set,
                                         bg='#0f3460', fg='#ffffff',
                                         font=('Arial', 9),
                                         selectmode='single',
                                         relief='solid', bd=2,
                                         selectbackground='#00d4ff',
                                         selectforeground='#000000',
                                         height=4)
        self.folder_listbox.pack(fill='both', expand=True)
        folder_scrollbar.config(command=self.folder_listbox.yview)
        
        folder_btn_frame = tk.Frame(folder_section, bg='#16213e')
        folder_btn_frame.pack(fill='x', pady=(5, 0))
        
        add_folder_btn = tk.Button(folder_btn_frame, text="➕ افزودن پوشه", 
                                   command=self.add_folder,
                                   bg='#00d4ff', fg='#000000', 
                                   font=('Arial', 9, 'bold'),
                                   relief='raised', bd=2, padx=10, pady=5,
                                   cursor='hand2')
        add_folder_btn.pack(side='left', padx=2, expand=True, fill='x')
        
        remove_folder_btn = tk.Button(folder_btn_frame, text="➖ حذف پوشه", 
                                      command=self.remove_folder,
                                      bg='#e74c3c', fg='#ffffff', 
                                      font=('Arial', 9, 'bold'),
                                      relief='raised', bd=2, padx=10, pady=5,
                                      cursor='hand2')
        remove_folder_btn.pack(side='left', padx=2, expand=True, fill='x')
        
        clear_folders_btn = tk.Button(folder_btn_frame, text="🗑️ پاک کردن همه", 
                                      command=self.clear_folders,
                                      bg='#95a5a6', fg='#ffffff', 
                                      font=('Arial', 9, 'bold'),
                                      relief='raised', bd=2, padx=10, pady=5,
                                      cursor='hand2')
        clear_folders_btn.pack(side='left', padx=2, expand=True, fill='x')
        
        settings_frame = tk.LabelFrame(parent, text="⚙️ تنظیمات", 
                                       bg='#16213e', fg='#00d4ff',
                                       font=('Arial', 10, 'bold'), 
                                       padx=10, pady=10, relief='solid', bd=2)
        settings_frame.pack(fill='x', pady=8)
        
        threshold_frame = tk.Frame(settings_frame, bg='#16213e')
        threshold_frame.pack(fill='x', pady=5)
        
        tk.Label(threshold_frame, text="آستانه:", 
                bg='#16213e', fg='#ffffff', 
                font=('Arial', 9)).pack(side='left', padx=3)
        
        self.similarity_threshold = tk.DoubleVar(value=0.97)
        threshold_scale = tk.Scale(threshold_frame, from_=0.85, to=0.995, 
                                  resolution=0.005,
                                  orient='horizontal', 
                                  variable=self.similarity_threshold,
                                  bg='#16213e', fg='#00ff88', 
                                  troughcolor='#0f3460', 
                                  highlightthickness=0,
                                  length=180, sliderlength=20)
        threshold_scale.pack(side='left', padx=5)
        
        self.threshold_value_label = tk.Label(threshold_frame, 
                                             text=f"{self.similarity_threshold.get():.3f}",
                                             bg='#16213e', fg='#00ff88',
                                             font=('Arial', 9, 'bold'))
        self.threshold_value_label.pack(side='left', padx=5)
        
        threshold_scale.config(command=self.update_threshold_label)
        
        batch_frame = tk.Frame(settings_frame, bg='#16213e')
        batch_frame.pack(fill='x', pady=5)
        
        tk.Label(batch_frame, text="Batch:", 
                bg='#16213e', fg='#ffffff', 
                font=('Arial', 9)).pack(side='left', padx=3)
        
        self.batch_size = tk.IntVar(value=32)
        batch_spinbox = tk.Spinbox(batch_frame, from_=8, to=128, 
                                   increment=8, textvariable=self.batch_size,
                                   width=8, font=('Arial', 9),
                                   bg='#0f3460', fg='#ffffff')
        batch_spinbox.pack(side='left', padx=5)
        
        self.include_subfolders = tk.BooleanVar(value=True)
        tk.Checkbutton(settings_frame, text="زیرپوشه‌ها", 
                      variable=self.include_subfolders,
                      bg='#16213e', fg='#ffffff', 
                      selectcolor='#0f3460',
                      font=('Arial', 9)).pack(anchor='w', pady=2)
        
        scan_btn = tk.Button(parent, text="🔍 شروع مقایسه", 
                            command=self.start_comparison,
                            bg='#00ff88', fg='#000000', 
                            font=('Arial', 11, 'bold'),
                            relief='raised', bd=3, pady=8,
                            cursor='hand2')
        scan_btn.pack(fill='x', pady=10)
        
        self.progress = ttk.Progressbar(parent, mode='determinate')
        self.progress.pack(fill='x', pady=5)
        
        self.progress_label = tk.Label(parent, text="", 
                                       bg='#1a1a2e', fg='#00d4ff', 
                                       font=('Arial', 8))
        self.progress_label.pack()
        
        self.status_label = tk.Label(parent, text="آماده", 
                                     bg='#1a1a2e', fg='#00ff88', 
                                     font=('Arial', 9, 'bold'))
        self.status_label.pack(pady=5)
        
        results_frame = tk.LabelFrame(parent, text="📊 گروه‌های مشابه", 
                                      bg='#16213e', fg='#00d4ff',
                                      font=('Arial', 10, 'bold'),
                                      relief='solid', bd=2)
        results_frame.pack(fill='both', expand=True, pady=8)
        
        scrollbar = tk.Scrollbar(results_frame)
        scrollbar.pack(side='right', fill='y')
        
        self.results_listbox = tk.Listbox(results_frame, 
                                          yscrollcommand=scrollbar.set,
                                          bg='#0f3460', fg='#ffffff',
                                          font=('Arial', 9),
                                          selectmode='single',
                                          relief='solid', bd=2,
                                          selectbackground='#00d4ff',
                                          selectforeground='#000000')
        self.results_listbox.pack(fill='both', expand=True, padx=3, pady=3)
        self.results_listbox.bind('<<ListboxSelect>>', self.on_group_select)
        scrollbar.config(command=self.results_listbox.yview)
        
        export_btn = tk.Button(parent, text="💾 ذخیره", 
                              command=self.export_results,
                              bg='#e74c3c', fg='#ffffff', 
                              font=('Arial', 10, 'bold'),
                              relief='raised', bd=2, pady=6,
                              cursor='hand2')
        export_btn.pack(fill='x', pady=5)
    
    def update_threshold_label(self, value):
        self.threshold_value_label.config(text=f"{float(value):.3f}")
    
    def add_folder(self):
        folder = filedialog.askdirectory(title="انتخاب پوشه")
        if folder and folder not in self.folder_paths:
            self.folder_paths.append(folder)
            folder_name = os.path.basename(folder) or folder
            self.folder_listbox.insert(tk.END, folder_name)
    
    def remove_folder(self):
        selection = self.folder_listbox.curselection()
        if selection:
            idx = selection[0]
            self.folder_listbox.delete(idx)
            self.folder_paths.pop(idx)
    
    def clear_folders(self):
        self.folder_listbox.delete(0, tk.END)
        self.folder_paths.clear()
    
    def setup_right_panel(self, parent):
        title_frame = tk.Frame(parent, bg='#16213e')
        title_frame.pack(fill='x', padx=10, pady=10)
        
        tk.Label(title_frame, text="🖼️ نمایش تصاویر", 
                font=('Arial', 14, 'bold'), bg='#16213e', fg='#00d4ff').pack()
        
        self.group_info_label = tk.Label(title_frame, text="گروهی انتخاب نشده", 
                                        font=('Arial', 9), bg='#16213e', fg='#a0a0a0')
        self.group_info_label.pack(pady=3)
        
        canvas_frame = tk.Frame(parent, bg='#16213e')
        canvas_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.image_canvas = tk.Canvas(canvas_frame, bg='#0f3460', 
                                      highlightthickness=0)
        scrollbar = tk.Scrollbar(canvas_frame, orient='vertical', 
                                command=self.image_canvas.yview)
        
        self.scrollable_frame = tk.Frame(self.image_canvas, bg='#0f3460')
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.image_canvas.configure(scrollregion=self.image_canvas.bbox("all"))
        )
        
        self.image_canvas.create_window((0, 0), window=self.scrollable_frame, 
                                       anchor="nw")
        self.image_canvas.configure(yscrollcommand=scrollbar.set)
        
        self.image_canvas.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        self.image_canvas.bind_all("<MouseWheel>", self._on_mousewheel)
    
    def _on_mousewheel(self, event):
        self.image_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    
    def start_comparison(self):
        if not self.folder_paths:
            messagebox.showwarning("هشدار", "لطفاً حداقل یک پوشه انتخاب کنید")
            return
        
        if self.model is None:
            messagebox.showerror("خطا", "مدل بارگذاری نشده است")
            return
        
        self.results_listbox.delete(0, tk.END)
        self.clear_image_display()
        self.progress['value'] = 0
        self.status_label.config(text="در حال پردازش...")
        
        thread = threading.Thread(target=self.compare_images_gpu)
        thread.daemon = True
        thread.start()
    
    def extract_features_batch(self, image_paths, batch_size):
        """Extract advanced perceptual features with multi-scale analysis"""
        features_dict = {}
        
        # تبدیلات پیشرفته‌تر با augmentation
        transform = transforms.Compose([
            transforms.Resize(380),  # سایز بزرگتر برای دقت بیشتر
            transforms.CenterCrop(380),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225]),
        ])
        
        # تبدیل کوچکتر برای multi-scale features
        transform_small = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225]),
        ])
        
        total_batches = (len(image_paths) + batch_size - 1) // batch_size
        
        for batch_idx in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[batch_idx:batch_idx + batch_size]
            batch_tensors_large = []
            batch_tensors_small = []
            valid_paths = []
            
            for img_path in batch_paths:
                try:
                    img = Image.open(img_path).convert('RGB')
                    
                    # استخراج features در دو مقیاس مختلف
                    img_tensor_large = transform(img)
                    img_tensor_small = transform_small(img)
                    
                    batch_tensors_large.append(img_tensor_large)
                    batch_tensors_small.append(img_tensor_small)
                    valid_paths.append(img_path)
                except Exception as e:
                    print(f"خطا در بارگذاری {img_path}: {e}")
            
            if batch_tensors_large:
                # پردازش هر دو مقیاس
                batch_tensor_large = torch.stack(batch_tensors_large).to(self.device)
                batch_tensor_small = torch.stack(batch_tensors_small).to(self.device)
                
                with torch.no_grad():
                    # استخراج features از هر دو مقیاس
                    features_large = self.model(batch_tensor_large)
                    features_small = self.model(batch_tensor_small)
                
                # ترکیب features از هر دو مقیاس برای دقت بیشتر
                features_large_np = features_large.cpu().numpy()
                features_small_np = features_small.cpu().numpy()
                
                for i, path in enumerate(valid_paths):
                    # ترکیب weighted از features
                    combined_features = np.concatenate([
                        features_large_np[i].flatten() * 0.7,  # وزن بیشتر برای سایز بزرگ
                        features_small_np[i].flatten() * 0.3   # وزن کمتر برای سایز کوچک
                    ])
                    features_dict[path] = combined_features
            
            progress = ((batch_idx // batch_size) + 1) / total_batches * 50
            self.root.after(0, lambda p=progress: self.progress.config(value=p))
            self.root.after(0, lambda: self.progress_label.config(
                text=f"استخراج پیشرفته: {len(features_dict)}/{len(image_paths)}"))
        
        return features_dict
    
    def compare_images_gpu(self):
        start_time = time.time()
        
        try:
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', 
                              '.tiff', '.webp', '.JPG', '.JPEG', '.PNG'}
            images = []
            
            self.root.after(0, lambda: self.status_label.config(
                text="🔍 جستجوی تصاویر..."))
            
            for folder_path in self.folder_paths:
                if self.include_subfolders.get():
                    for root, dirs, files in os.walk(folder_path):
                        for file in files:
                            if Path(file).suffix in image_extensions:
                                images.append(os.path.join(root, file))
                else:
                    for f in os.listdir(folder_path):
                        if Path(f).suffix in image_extensions:
                            images.append(os.path.join(folder_path, f))
            
            if len(images) < 2:
                self.root.after(0, lambda: messagebox.showinfo("اطلاعات", 
                    "حداقل دو تصویر نیاز است"))
                return
            
            self.root.after(0, lambda: self.status_label.config(
                text=f"📸 {len(images)} تصویر - پردازش..."))
            
            batch_size = self.batch_size.get()
            features_dict = self.extract_features_batch(images, batch_size)
            
            self.root.after(0, lambda: self.status_label.config(
                text="🧮 محاسبه تشابه..."))
            self.root.after(0, lambda: self.progress.config(value=60))
            
            paths = list(features_dict.keys())
            features = np.array([features_dict[p] for p in paths])
            
            # نرمال‌سازی features برای دقت بیشتر
            from sklearn.preprocessing import normalize
            features = normalize(features, norm='l2')
            
            # محاسبه تشابه با cosine similarity
            similarity_matrix = cosine_similarity(features)
            
            self.root.after(0, lambda: self.progress.config(value=80))
            
            threshold = self.similarity_threshold.get()
            self.duplicates = []
            processed = set()
            
            # الگوریتم پیشرفته‌تر برای گروه‌بندی
            for i in range(len(paths)):
                if i in processed:
                    continue
                
                group = [paths[i]]
                similarities = []
                
                for j in range(i + 1, len(paths)):
                    if j not in processed:
                        sim_score = similarity_matrix[i][j]
                        
                        # بررسی تشابه با تمام اعضای گروه (نه فقط اولین عکس)
                        # این باعث می‌شود فقط عکس‌های واقعاً مشابه گروه‌بندی شوند
                        if len(group) == 1:
                            # اولین مقایسه
                            if sim_score >= threshold:
                                group.append(paths[j])
                                similarities.append(sim_score)
                                processed.add(j)
                        else:
                            # بررسی تشابه با حداقل 70% از اعضای گروه
                            similar_count = 0
                            temp_similarities = []
                            
                            for k, member in enumerate(group):
                                member_idx = paths.index(member)
                                member_sim = similarity_matrix[member_idx][j]
                                if member_sim >= threshold:
                                    similar_count += 1
                                    temp_similarities.append(member_sim)
                            
                            # اگر با حداقل 70% از گروه مشابه بود، اضافه کن
                            if similar_count / len(group) >= 0.7:
                                group.append(paths[j])
                                similarities.append(np.mean(temp_similarities))
                                processed.add(j)
                
                if len(group) > 1:
                    self.duplicates.append({
                        'paths': group,
                        'similarity': similarities
                    })
                    processed.add(i)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            self.root.after(0, lambda: self.display_results(processing_time))
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("خطا", 
                f"خطا: {str(e)}"))
        finally:
            self.root.after(0, lambda: self.progress.config(value=100))
    
    def display_results(self, processing_time):
        self.results_listbox.delete(0, tk.END)
        
        if not self.duplicates:
            self.status_label.config(
                text=f"✅ کامل شد در {processing_time:.1f}s - تصویر مشابه نیافت")
            self.results_listbox.insert(tk.END, "تصویر مشابهی نیافت")
        else:
            total_images = sum(len(g['paths']) for g in self.duplicates)
            self.status_label.config(
                text=f"✅ {len(self.duplicates)} گروه ({total_images} تصویر) - {processing_time:.1f}s")
            
            for idx, group_data in enumerate(self.duplicates):
                paths = group_data['paths']
                avg_sim = np.mean(group_data['similarity']) * 100 if group_data['similarity'] else 0
                
                display_text = f"گروه {idx+1}: {len(paths)} تصویر (تشابه: {avg_sim:.1f}%)"
                self.results_listbox.insert(tk.END, display_text)
    
    def on_group_select(self, event):
        selection = self.results_listbox.curselection()
        if not selection:
            return
        
        idx = selection[0]
        if idx >= len(self.duplicates):
            return
        
        self.current_group = self.duplicates[idx]
        self.display_images()
    
    def clear_image_display(self):
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        self.photo_references.clear()
        self.group_info_label.config(text="گروهی انتخاب نشده")
    
    def create_rounded_image(self, img, size=(350, 280)):
        """Create a rounded corner image"""
        img.thumbnail(size, Image.Resampling.LANCZOS)
        
        mask = Image.new('L', img.size, 0)
        draw = ImageDraw.Draw(mask)
        draw.rounded_rectangle([(0, 0), img.size], radius=15, fill=255)
        
        output = Image.new('RGBA', img.size, (0, 0, 0, 0))
        output.paste(img, (0, 0))
        output.putalpha(mask)
        
        return output
    
    def copy_to_clipboard(self, text):
        """Copy text to clipboard"""
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.root.update()
        messagebox.showinfo("کپی شد", f"متن کپی شد:\n{text}")
    
    def display_images(self):
        self.clear_image_display()
        
        if not self.current_group:
            return
        
        paths = self.current_group['paths']
        similarities = self.current_group['similarity']
        
        avg_sim = np.mean(similarities) * 100 if similarities else 0
        self.group_info_label.config(
            text=f"{len(paths)} تصویر در این گروه | میانگین تشابه: {avg_sim:.1f}%"
        )
        
        cols = 2
        
        for idx, img_path in enumerate(paths):
            try:
                row = idx // cols
                col = idx % cols
                
                card_outer = tk.Frame(self.scrollable_frame, bg='#0f3460', 
                                     relief='flat', bd=0)
                card_outer.grid(row=row, column=col, padx=15, pady=15, sticky='nsew')
                
                card_frame = tk.Frame(card_outer, bg='#1a2332', 
                                     relief='solid', bd=0)
                card_frame.pack(padx=3, pady=3, fill='both', expand=True)
                
                header_frame = tk.Frame(card_frame, bg='#0d1b2a', height=50)
                header_frame.pack(fill='x', padx=0, pady=0)
                header_frame.pack_propagate(False)
                
                badge_frame = tk.Frame(header_frame, bg='#0d1b2a')
                badge_frame.pack(fill='both', expand=True, padx=10, pady=8)
                
                if idx == 0:
                    badge = tk.Label(badge_frame, text="⭐ مرجع", 
                                   bg='#00d4ff', fg='#000000',
                                   font=('Arial', 8, 'bold'),
                                   padx=8, pady=3, relief='flat')
                    badge.pack(side='left')
                else:
                    sim_percent = similarities[idx-1] * 100
                    badge_color = '#00ff88' if sim_percent >= 97 else '#ffd700' if sim_percent >= 95 else '#ff6b6b'
                    badge = tk.Label(badge_frame, text=f"🔗 {sim_percent:.1f}%", 
                                   bg=badge_color, fg='#000000',
                                   font=('Arial', 8, 'bold'),
                                   padx=8, pady=3, relief='flat')
                    badge.pack(side='left')
                
                file_num = tk.Label(badge_frame, text=f"#{idx+1}", 
                                  bg='#0d1b2a', fg='#7f8c8d',
                                  font=('Arial', 9, 'bold'))
                file_num.pack(side='right')
                
                img_container = tk.Frame(card_frame, bg='#1a2332', 
                                        relief='flat', bd=0)
                img_container.pack(padx=10, pady=10)
                
                img = Image.open(img_path).convert('RGB')
                rounded_img = self.create_rounded_image(img, (350, 280))
                photo = ImageTk.PhotoImage(rounded_img)
                self.photo_references.append(photo)
                
                img_label = tk.Label(img_container, image=photo, 
                                   bg='#1a2332', relief='flat', bd=0)
                img_label.pack()
                
                filename = os.path.basename(img_path)
                display_filename = filename if len(filename) <= 35 else filename[:32] + "..."
                
                filename_frame = tk.Frame(card_frame, bg='#1a2332', cursor='hand2')
                filename_frame.pack(fill='x', padx=10, pady=(0, 5))
                
                # کلیک برای کپی کردن نام فایل
                filename_frame.bind('<Button-1>', lambda e, fn=filename: self.copy_to_clipboard(fn))
                
                icon_label = tk.Label(filename_frame, text="📄", 
                        bg='#1a2332', fg='#00d4ff',
                        font=('Arial', 10), cursor='hand2')
                icon_label.pack(side='left', padx=(5, 3))
                icon_label.bind('<Button-1>', lambda e, fn=filename: self.copy_to_clipboard(fn))
                
                name_label = tk.Label(filename_frame, text=display_filename, 
                        bg='#1a2332', fg='#ffffff',
                        font=('Arial', 9, 'bold'),
                        anchor='w', cursor='hand2')
                name_label.pack(side='left', fill='x', expand=True)
                name_label.bind('<Button-1>', lambda e, fn=filename: self.copy_to_clipboard(fn))
                
                # تغییر رنگ هنگام هاور
                def on_enter(e):
                    name_label.config(fg='#00d4ff')
                    icon_label.config(fg='#00ff88')
                
                def on_leave(e):
                    name_label.config(fg='#ffffff')
                    icon_label.config(fg='#00d4ff')
                
                filename_frame.bind('<Enter>', on_enter)
                filename_frame.bind('<Leave>', on_leave)
                icon_label.bind('<Enter>', on_enter)
                icon_label.bind('<Leave>', on_leave)
                name_label.bind('<Enter>', on_enter)
                name_label.bind('<Leave>', on_leave)
                
                try:
                    file_size = os.path.getsize(img_path)
                    size_mb = file_size / (1024 * 1024)
                    img_obj = Image.open(img_path)
                    dimensions = f"{img_obj.width}x{img_obj.height}"
                    
                    info_frame = tk.Frame(card_frame, bg='#0d1b2a')
                    info_frame.pack(fill='x', padx=10, pady=(0, 10))
                    
                    tk.Label(info_frame, text=f"📐 {dimensions}", 
                            bg='#0d1b2a', fg='#7f8c8d',
                            font=('Arial', 8)).pack(side='left', padx=5)
                    
                    tk.Label(info_frame, text=f"💾 {size_mb:.2f} MB", 
                            bg='#0d1b2a', fg='#7f8c8d',
                            font=('Arial', 8)).pack(side='left', padx=5)
                except:
                    pass
                
                btn_frame = tk.Frame(card_frame, bg='#1a2332')
                btn_frame.pack(fill='x', padx=10, pady=(0, 10))
                
                open_btn = tk.Button(btn_frame, text="📁 باز کردن", 
                                    command=lambda p=img_path: self.open_folder(p),
                                    bg='#3498db', fg='#ffffff',
                                    font=('Arial', 9, 'bold'),
                                    relief='flat', bd=0,
                                    cursor='hand2', padx=15, pady=8,
                                    activebackground='#2980b9',
                                    activeforeground='#ffffff')
                open_btn.pack(side='left', padx=3, expand=True, fill='x')
                
                delete_btn = tk.Button(btn_frame, text="🗑️ حذف", 
                                      command=lambda p=img_path: self.delete_image(p),
                                      bg='#e74c3c', fg='#ffffff',
                                      font=('Arial', 9, 'bold'),
                                      relief='flat', bd=0,
                                      cursor='hand2', padx=15, pady=8,
                                      activebackground='#c0392b',
                                      activeforeground='#ffffff')
                delete_btn.pack(side='left', padx=3, expand=True, fill='x')
                
                preview_btn = tk.Button(btn_frame, text="👁️ پیش‌نمایش", 
                                       command=lambda p=img_path: self.preview_image(p),
                                       bg='#9b59b6', fg='#ffffff',
                                       font=('Arial', 9, 'bold'),
                                       relief='flat', bd=0,
                                       cursor='hand2', padx=15, pady=8,
                                       activebackground='#8e44ad',
                                       activeforeground='#ffffff')
                preview_btn.pack(side='left', padx=3, expand=True, fill='x')
                
                def on_enter(e, btn, color):
                    btn['background'] = color
                
                def on_leave(e, btn, color):
                    btn['background'] = color
                
                open_btn.bind("<Enter>", lambda e: on_enter(e, open_btn, '#2980b9'))
                open_btn.bind("<Leave>", lambda e: on_leave(e, open_btn, '#3498db'))
                
                delete_btn.bind("<Enter>", lambda e: on_enter(e, delete_btn, '#c0392b'))
                delete_btn.bind("<Leave>", lambda e: on_leave(e, delete_btn, '#e74c3c'))
                
                preview_btn.bind("<Enter>", lambda e: on_enter(e, preview_btn, '#8e44ad'))
                preview_btn.bind("<Leave>", lambda e: on_leave(e, preview_btn, '#9b59b6'))
                
                # مسیر کامل فایل - قابل کپی
                path_frame = tk.Frame(card_frame, bg='#0d1b2a', cursor='hand2')
                path_frame.pack(fill='x', padx=0, pady=0)
                path_frame.bind('<Button-1>', lambda e, p=img_path: self.copy_to_clipboard(p))
                
                path_label = tk.Label(path_frame, text=img_path, 
                        bg='#0d1b2a', fg='#5a6c7d',
                        font=('Arial', 7),
                        anchor='w', wraplength=360, 
                        justify='left', cursor='hand2')
                path_label.pack(fill='x', padx=10, pady=5)
                path_label.bind('<Button-1>', lambda e, p=img_path: self.copy_to_clipboard(p))
                
                # تغییر رنگ هنگام هاور روی مسیر
                def path_on_enter(e):
                    path_label.config(fg='#00d4ff')
                
                def path_on_leave(e):
                    path_label.config(fg='#5a6c7d')
                
                path_frame.bind('<Enter>', path_on_enter)
                path_frame.bind('<Leave>', path_on_leave)
                path_label.bind('<Enter>', path_on_enter)
                path_label.bind('<Leave>', path_on_leave)
                
            except Exception as e:
                print(f"خطا در نمایش {img_path}: {e}")
        
        for i in range(cols):
            self.scrollable_frame.grid_columnconfigure(i, weight=1)
    
    def preview_image(self, file_path):
        """Show full size image preview in new window"""
        try:
            preview_window = tk.Toplevel(self.root)
            preview_window.title("پیش‌نمایش تصویر")
            preview_window.configure(bg='#1a1a2e')
            
            window_width = 800
            window_height = 600
            screen_width = preview_window.winfo_screenwidth()
            screen_height = preview_window.winfo_screenheight()
            x = (screen_width - window_width) // 2
            y = (screen_height - window_height) // 2
            preview_window.geometry(f"{window_width}x{window_height}+{x}+{y}")
            
            header = tk.Frame(preview_window, bg='#16213e')
            header.pack(fill='x', padx=10, pady=10)
            
            filename = os.path.basename(file_path)
            tk.Label(header, text=f"📷 {filename}", 
                    bg='#16213e', fg='#00d4ff',
                    font=('Arial', 12, 'bold')).pack()
            
            img_frame = tk.Frame(preview_window, bg='#0f3460')
            img_frame.pack(fill='both', expand=True, padx=10, pady=10)
            
            img = Image.open(file_path)
            
            max_width = 750
            max_height = 500
            img.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
            
            photo = ImageTk.PhotoImage(img)
            
            img_label = tk.Label(img_frame, image=photo, bg='#0f3460')
            img_label.image = photo
            img_label.pack(expand=True)
            
            close_btn = tk.Button(preview_window, text="بستن", 
                                 command=preview_window.destroy,
                                 bg='#e74c3c', fg='#ffffff',
                                 font=('Arial', 10, 'bold'),
                                 relief='flat', bd=0,
                                 cursor='hand2', padx=20, pady=8)
            close_btn.pack(pady=10)
            
        except Exception as e:
            messagebox.showerror("خطا", f"خطا در نمایش تصویر: {str(e)}")
    
    def open_folder(self, file_path):
        """Open file location in file explorer - FIXED VERSION"""
        try:
            # اطمینان از اینکه مسیر فایل صحیح است
            file_path = os.path.abspath(file_path)
            folder_path = os.path.dirname(file_path)
            
            system = platform.system()
            if system == 'Windows':
                # در ویندوز از explorer با /select استفاده می‌کنیم
                subprocess.Popen(['explorer', '/select,', os.path.normpath(file_path)])
            elif system == 'Darwin':  # macOS
                subprocess.run(['open', '-R', file_path])
            else:  # Linux
                # در لینوکس فقط پوشه را باز می‌کنیم
                subprocess.run(['xdg-open', folder_path])
                
        except Exception as e:
            messagebox.showerror("خطا", f"خطا در باز کردن پوشه: {str(e)}")
    
    def delete_image(self, file_path):
        """Delete image file"""
        result = messagebox.askyesno("تأیید حذف", 
            f"آیا مطمئنید که می‌خواهید این فایل را حذف کنید?\n\n{os.path.basename(file_path)}")
        
        if result:
            try:
                os.remove(file_path)
                messagebox.showinfo("موفقیت", "فایل با موفقیت حذف شد")
                
                if self.current_group:
                    self.current_group['paths'].remove(file_path)
                    
                    if len(self.current_group['paths']) < 2:
                        self.duplicates.remove(self.current_group)
                        self.current_group = None
                        self.clear_image_display()
                        
                        self.results_listbox.delete(0, tk.END)
                        for idx, group_data in enumerate(self.duplicates):
                            paths = group_data['paths']
                            avg_sim = np.mean(group_data['similarity']) * 100 if group_data['similarity'] else 0
                            display_text = f"گروه {idx+1}: {len(paths)} تصویر ({avg_sim:.1f}%)"
                            self.results_listbox.insert(tk.END, display_text)
                    else:
                        self.display_images()
                
            except Exception as e:
                messagebox.showerror("خطا", f"خطا در حذف فایل: {str(e)}")
    
    def export_results(self):
        if not self.duplicates:
            messagebox.showinfo("اطلاعات", "نتیجه‌ای وجود ندارد")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write("نتایج مقایسه تصاویر\n")
                    f.write("="*70 + "\n\n")
                    
                    f.write(f"تعداد پوشه‌های اسکن شده: {len(self.folder_paths)}\n")
                    for folder in self.folder_paths:
                        f.write(f"  - {folder}\n")
                    f.write("\n")
                    
                    for idx, group_data in enumerate(self.duplicates, 1):
                        paths = group_data['paths']
                        similarities = group_data['similarity']
                        
                        avg_sim = np.mean(similarities) * 100 if similarities else 0
                        
                        f.write(f"گروه {idx} ({len(paths)} تصویر - میانگین تشابه: {avg_sim:.2f}%):\n")
                        f.write("-"*70 + "\n")
                        f.write(f"  مرجع: {paths[0]}\n\n")
                        
                        for i, img_path in enumerate(paths[1:]):
                            sim = similarities[i] * 100
                            f.write(f"  {sim:.2f}%: {img_path}\n")
                        
                        f.write("\n")
                
                messagebox.showinfo("موفقیت", "نتایج ذخیره شد")
            except Exception as e:
                messagebox.showerror("خطا", f"خطا: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = AdvancedImageComparator(root)
    root.mainloop()