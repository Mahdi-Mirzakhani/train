import cv2
import numpy as np
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import random

class YOLOLabelViewer:
    def __init__(self):
        """YOLO Label Viewer for checking dataset"""
        self.dataset_path = None
        self.image_files = []
        self.current_index = 0
        self.current_image = None
        self.class_names = {}
        self.colors = {}
        
        # Setup GUI
        self.setup_gui()
        
    def setup_gui(self):
        """Setup the main GUI window"""
        self.root = tk.Tk()
        self.root.title("🎯 YOLO Label Viewer")
        self.root.geometry("1200x800")
        self.root.configure(bg='#2b2b2b')
        
        # Main frame
        main_frame = tk.Frame(self.root, bg='#2b2b2b')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Control panel
        control_frame = tk.Frame(main_frame, bg='#3b3b3b', relief=tk.RAISED, bd=2)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Browse button
        tk.Button(
            control_frame, 
            text="📂 Select Dataset Folder", 
            command=self.browse_dataset,
            bg='#4CAF50', fg='white', font=('Arial', 12, 'bold'),
            padx=20, pady=5
        ).pack(side=tk.LEFT, padx=10, pady=10)
        
        # Navigation buttons
        nav_frame = tk.Frame(control_frame, bg='#3b3b3b')
        nav_frame.pack(side=tk.RIGHT, padx=10, pady=10)
        
        tk.Button(
            nav_frame, 
            text="⬅️ Previous", 
            command=self.prev_image,
            bg='#2196F3', fg='white', font=('Arial', 10, 'bold'),
            padx=15, pady=5
        ).pack(side=tk.LEFT, padx=(0, 5))
        
        tk.Button(
            nav_frame, 
            text="Next ➡️", 
            command=self.next_image,
            bg='#2196F3', fg='white', font=('Arial', 10, 'bold'),
            padx=15, pady=5
        ).pack(side=tk.LEFT, padx=(5, 0))
        
        # Info panel
        info_frame = tk.Frame(main_frame, bg='#3b3b3b', relief=tk.RAISED, bd=2)
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.info_label = tk.Label(
            info_frame, 
            text="📁 No dataset loaded", 
            bg='#3b3b3b', fg='white', 
            font=('Arial', 11), pady=10
        )
        self.info_label.pack()
        
        # Image display area
        image_frame = tk.Frame(main_frame, bg='#2b2b2b', relief=tk.SUNKEN, bd=2)
        image_frame.pack(fill=tk.BOTH, expand=True)
        
        # Canvas for image display
        self.canvas = tk.Canvas(image_frame, bg='#1a1a1a', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Scrollbars
        h_scroll = tk.Scrollbar(image_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        v_scroll = tk.Scrollbar(image_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=h_scroll.set, yscrollcommand=v_scroll.set)
        
        # Class legend frame
        legend_frame = tk.Frame(main_frame, bg='#3b3b3b', relief=tk.RAISED, bd=2)
        legend_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.legend_label = tk.Label(
            legend_frame, 
            text="🏷️ Classes will appear here", 
            bg='#3b3b3b', fg='white', 
            font=('Arial', 10), pady=5
        )
        self.legend_label.pack()
        
        # Keyboard bindings
        self.root.bind('<Left>', lambda e: self.prev_image())
        self.root.bind('<Right>', lambda e: self.next_image())
        self.root.bind('<space>', lambda e: self.next_image())
        self.root.focus_set()
        
    def browse_dataset(self):
        """Browse and select dataset folder"""
        folder = filedialog.askdirectory(title="📂 Select Dataset Folder")
        if folder:
            self.dataset_path = Path(folder)
            self.load_dataset()
    
    def load_dataset(self):
        """Load dataset from selected folder"""
        try:
            print(f"📂 Loading dataset from: {self.dataset_path}")
            
            # Find all image files
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
            self.image_files = []
            
            for ext in image_extensions:
                self.image_files.extend(list(self.dataset_path.glob(f"*{ext}")))
                self.image_files.extend(list(self.dataset_path.glob(f"*{ext.upper()}")))
            
            if not self.image_files:
                messagebox.showerror("Error", "No image files found in the selected folder!")
                return
            
            # Sort files
            self.image_files.sort()
            
            # Load class names from YAML if exists
            self.load_class_names()
            
            # Generate colors for classes
            self.generate_colors()
            
            # Reset index and display first image
            self.current_index = 0
            self.display_current_image()
            
            print(f"✅ Loaded {len(self.image_files)} images")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error loading dataset:\n{str(e)}")
    
    def load_class_names(self):
        """Load class names from YAML file if exists"""
        try:
            yaml_files = list(self.dataset_path.glob("*.yaml"))
            if yaml_files:
                import yaml
                with open(yaml_files[0], 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                    if 'names' in config:
                        if isinstance(config['names'], list):
                            self.class_names = {i: name for i, name in enumerate(config['names'])}
                        else:
                            self.class_names = config['names']
                print(f"📋 Loaded class names: {list(self.class_names.values())}")
        except Exception as e:
            print(f"⚠️ Could not load class names: {e}")
            self.class_names = {}
    
    def generate_colors(self):
        """Generate random colors for each class"""
        if self.class_names:
            max_class = max(self.class_names.keys()) if self.class_names else 10
        else:
            max_class = 10
            
        self.colors = {}
        for i in range(max_class + 1):
            # Generate bright, distinct colors
            self.colors[i] = (
                random.randint(50, 255),
                random.randint(50, 255), 
                random.randint(50, 255)
            )
    
    def load_labels(self, image_path):
        """Load YOLO labels for given image"""
        label_path = image_path.with_suffix('.txt')
        
        if not label_path.exists():
            return []
        
        labels = []
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])
                            
                            labels.append({
                                'class_id': class_id,
                                'x_center': x_center,
                                'y_center': y_center,
                                'width': width,
                                'height': height
                            })
        except Exception as e:
            print(f"⚠️ Error reading label file {label_path}: {e}")
        
        return labels
    
    def draw_labels_on_image(self, image, labels):
        """Draw bounding boxes and labels on image"""
        if not labels:
            return image
            
        img_height, img_width = image.shape[:2]
        
        for label in labels:
            # Convert YOLO format to pixel coordinates
            x_center = label['x_center'] * img_width
            y_center = label['y_center'] * img_height
            width = label['width'] * img_width
            height = label['height'] * img_height
            
            # Calculate corner coordinates
            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)
            
            # Get class info
            class_id = label['class_id']
            class_name = self.class_names.get(class_id, f"Class_{class_id}")
            color = self.colors.get(class_id, (0, 255, 0))
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background
            label_text = f"{class_name} ({class_id})"
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            
            cv2.rectangle(
                image, 
                (x1, y1 - text_height - 10), 
                (x1 + text_width, y1), 
                color, -1
            )
            
            # Draw label text
            cv2.putText(
                image, label_text, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )
        
        return image
    
    def display_current_image(self):
        """Display current image with labels"""
        if not self.image_files or self.current_index >= len(self.image_files):
            return
        
        try:
            # Load image
            image_path = self.image_files[self.current_index]
            image = cv2.imread(str(image_path))
            
            if image is None:
                messagebox.showerror("Error", f"Cannot load image: {image_path.name}")
                return
            
            # Load and draw labels
            labels = self.load_labels(image_path)
            image_with_labels = self.draw_labels_on_image(image.copy(), labels)
            
            # Convert to RGB for display
            image_rgb = cv2.cvtColor(image_with_labels, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image and then to PhotoImage
            pil_image = Image.fromarray(image_rgb)
            
            # Resize image if too large
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            if canvas_width > 1 and canvas_height > 1:
                # Calculate scale to fit canvas
                scale_x = canvas_width / pil_image.width
                scale_y = canvas_height / pil_image.height
                scale = min(scale_x, scale_y, 1.0)  # Don't upscale
                
                if scale < 1.0:
                    new_width = int(pil_image.width * scale)
                    new_height = int(pil_image.height * scale)
                    pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            self.current_image = ImageTk.PhotoImage(pil_image)
            
            # Clear canvas and display image
            self.canvas.delete("all")
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
            self.canvas.create_image(
                self.canvas.winfo_width() // 2,
                self.canvas.winfo_height() // 2,
                anchor=tk.CENTER,
                image=self.current_image
            )
            
            # Update info
            self.update_info(image_path, labels)
            
            # Update legend
            self.update_legend(labels)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error displaying image:\n{str(e)}")
    
    def update_info(self, image_path, labels):
        """Update information panel"""
        info_text = (
            f"📁 Dataset: {self.dataset_path.name} | "
            f"📸 Image: {self.current_index + 1}/{len(self.image_files)} | "
            f"🏷️ Labels: {len(labels)} | "
            f"📄 File: {image_path.name}"
        )
        self.info_label.config(text=info_text)
    
    def update_legend(self, labels):
        """Update class legend"""
        if not labels:
            self.legend_label.config(text="🏷️ No labels in this image")
            return
        
        # Get unique classes in current image
        classes_in_image = list(set(label['class_id'] for label in labels))
        classes_in_image.sort()
        
        legend_text = "🏷️ Classes in image: "
        for class_id in classes_in_image:
            class_name = self.class_names.get(class_id, f"Class_{class_id}")
            count = sum(1 for label in labels if label['class_id'] == class_id)
            legend_text += f"[{class_id}:{class_name}×{count}] "
        
        self.legend_label.config(text=legend_text)
    
    def prev_image(self):
        """Go to previous image"""
        if self.image_files and self.current_index > 0:
            self.current_index -= 1
            self.display_current_image()
    
    def next_image(self):
        """Go to next image"""
        if self.image_files and self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self.display_current_image()
    
    def run(self):
        """Run the viewer"""
        print("🎯 YOLO Label Viewer")
        print("Controls:")
        print("  • Click 'Select Dataset Folder' to load dataset")
        print("  • Use ⬅️➡️ buttons or arrow keys to navigate")
        print("  • Press Space for next image")
        print("=" * 50)
        
        self.root.mainloop()

# Run the viewer
if __name__ == "__main__":
    try:
        viewer = YOLOLabelViewer()
        viewer.run()
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        input("Press Enter to exit...")