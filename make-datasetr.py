import cv2
import os
import torch
from pathlib import Path
import yaml
from ultralytics import YOLO
import shutil
import random
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import tkinter.simpledialog
from typing import Union
import sys

class YOLODatasetGenerator:
    def __init__(self):
        """YOLO Dataset Generator with GUI interface"""
        self.model = None
        self.input_path = None
        self.output_dir = None
        self.input_type = None
        self.root = None
        
        # ===== مسیرهای ثابت - اینجا مسیرهای خود را قرار دهید =====
        self.model_path = "D:\python\detect-face//ATRAS-l.pt"  # مسیر مدل خود را اینجا وارد کنید
        self.base_output_dir = "D:\\python\\dataset\\New folder"  # مسیر پوشه خروجی را اینجا وارد کنید
        # ================================================
        
    def select_paths_gui(self):
        """Select only input path through GUI interface"""
        self.root = tk.Tk()
        self.root.withdraw()  # Hide main window
        
        print("🎯 YOLO Dataset Generator")
        print("=" * 50)
        
        try:
            # Select input type
            choice = messagebox.askyesno(
                "Input Type", 
                "Select input type:\n\nYes = Image Folder\nNo = Video File"
            )
            
            if choice:  # Image folder
                self.input_path = filedialog.askdirectory(
                    title="📂 Select Image Folder"
                )
                self.input_type = "folder"
                print(f"📂 Selected image folder: {self.input_path}")
            else:  # Video file
                self.input_path = filedialog.askopenfilename(
                    title="🎬 Select Video File",
                    filetypes=[
                        ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv"),
                        ("All files", "*.*")
                    ]
                )
                self.input_type = "video"
                print(f"🎬 Selected video file: {self.input_path}")
            
            if not self.input_path:
                messagebox.showerror("Error", "No input selected!")
                return False
            
            # Setup output directory automatically
            input_name = Path(self.input_path).stem
            self.output_dir = Path(self.base_output_dir) / f"{input_name}_dataset"
            print(f"💾 Output directory: {self.output_dir}")
            
            # Check if model file exists
            if not Path(self.model_path).exists():
                messagebox.showerror("Error", f"Model file not found at:\n{self.model_path}\n\nPlease update the model_path in the code.")
                return False
            
            print(f"🤖 Using model: {self.model_path}")
            
            # Load model
            try:
                print("⏳ Loading model...")
                self.model = YOLO(self.model_path)
                print("✅ Model loaded successfully!")
                messagebox.showinfo("Success", f"Model loaded successfully!\n\nInput: {Path(self.input_path).name}\nOutput: {self.output_dir}")
            except Exception as e:
                messagebox.showerror("Error", f"Error loading model:\n{str(e)}")
                return False
            
            return True
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            return False
        
    def setup_directories(self):
        """Create directory structure - images and labels in same folder"""
        # Create base output directory if it doesn't exist
        Path(self.base_output_dir).mkdir(parents=True, exist_ok=True)
        
        # Create main dataset directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Directory structure created at {self.output_dir}")
    
    def find_all_images(self, folder_path):
        """جستجوی بازگشتی برای پیدا کردن تمام عکس‌ها در پوشه و زیرپوشه‌ها"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.tif'}
        image_paths = []
        
        print(f"🔍 Searching for images in: {folder_path}")
        
        # جستجوی بازگشتی در تمام پوشه‌ها
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = Path(root) / file
                if file_path.suffix.lower() in image_extensions:
                    image_paths.append(file_path)
        
        # نمایش آمار پوشه‌ها
        unique_folders = set()
        for img_path in image_paths:
            unique_folders.add(str(img_path.parent))
        
        print(f"📊 Search results:")
        print(f"   • Total images found: {len(image_paths)}")
        print(f"   • Folders searched: {len(unique_folders)}")
        
        # نمایش نمونه‌ای از پوشه‌های پیدا شده
        if len(unique_folders) > 0:
            print(f"   • Sample folders:")
            for i, folder in enumerate(sorted(unique_folders)[:5]):
                folder_name = Path(folder).name
                count = sum(1 for img in image_paths if str(img.parent) == folder)
                print(f"     - {folder_name}: {count} images")
            if len(unique_folders) > 5:
                print(f"     - ... and {len(unique_folders) - 5} more folders")
        
        return sorted(image_paths)
    
    def process_video(self, frame_interval=30):
        """Process video and extract frames"""
        print(f"\n🎬 Starting video processing: {self.input_path}")
        
        cap = cv2.VideoCapture(self.input_path)
        if not cap.isOpened():
            raise ValueError("Cannot open video file!")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"📊 Video information:")
        print(f"   • Total frames: {total_frames}")
        print(f"   • FPS: {fps:.2f}")
        print(f"   • Duration: {duration:.2f} seconds")
        print(f"   • Frames to extract: {total_frames // frame_interval}")
        
        frame_count = 0
        saved_count = 0
        video_name = Path(self.input_path).stem
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                # Predict with model
                results = self.model(frame, verbose=False)
                
                # Save frame and labels in same directory
                self.save_frame_and_labels(
                    frame, results[0], 
                    f"{video_name}_frame_{saved_count:06d}"
                )
                saved_count += 1
                
                if saved_count % 50 == 0:
                    print(f"⏳ Processed: {saved_count} frames")
                
            frame_count += 1
            
        cap.release()
        print(f"✅ {saved_count} frames processed from video")
        return saved_count
    
    def process_image_folder(self):
        """Process folder containing images - با جستجوی بازگشتی"""
        print(f"\n📂 Starting recursive folder processing: {self.input_path}")
        
        # جستجوی بازگشتی برای پیدا کردن تمام عکس‌ها
        image_paths = self.find_all_images(self.input_path)
        
        if not image_paths:
            print("❌ No images found in folder and subfolders!")
            messagebox.showwarning("No Images", "No images found in the selected folder and its subfolders!")
            return 0
        
        # تأیید از کاربر برای ادامه
        confirm = messagebox.askyesno(
            "Confirm Processing",
            f"Found {len(image_paths)} images in folder and subfolders.\n\nDo you want to continue processing all of them?",
            icon="question"
        )
        
        if not confirm:
            print("❌ Processing cancelled by user")
            return 0
        
        processed_count = 0
        failed_count = 0
        
        for i, img_path in enumerate(image_paths):
            try:
                # Read image
                frame = cv2.imread(str(img_path))
                if frame is None:
                    print(f"⚠️ Cannot read image {img_path.name}")
                    failed_count += 1
                    continue
                    
                # Predict with model
                results = self.model(frame, verbose=False)
                
                # ایجاد نام منحصر به فرد برای فایل (شامل نام پوشه والد)
                parent_folder = img_path.parent.name
                unique_name = f"{parent_folder}_{img_path.stem}"
                
                # در صورت تکراری بودن، شماره اضافه کن
                counter = 1
                original_name = unique_name
                while (self.output_dir / f"{unique_name}.jpg").exists():
                    unique_name = f"{original_name}_{counter}"
                    counter += 1
                
                # Save image and labels in same directory
                self.save_frame_and_labels(
                    frame, results[0],
                    unique_name
                )
                
                processed_count += 1
                
                # نمایش پیشرفت
                if processed_count % 50 == 0:
                    print(f"⏳ Processed: {processed_count}/{len(image_paths)} images")
                    
            except Exception as e:
                print(f"❌ Error processing {img_path.name}: {str(e)}")
                failed_count += 1
                continue
            
        print(f"✅ Processing completed!")
        print(f"   • Successfully processed: {processed_count} images")
        if failed_count > 0:
            print(f"   • Failed to process: {failed_count} images")
            
        return processed_count
    
    def save_frame_and_labels(self, frame, result, name):
        """Save frame and its labels in the same directory"""
        # Save image directly in output directory
        img_path = self.output_dir / f"{name}.jpg"
        cv2.imwrite(str(img_path), frame)
        
        # Save label in same directory
        label_path = self.output_dir / f"{name}.txt"
        
        h, w = frame.shape[:2]
        
        with open(label_path, 'w') as f:
            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    # Extract coordinates and class
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    conf = float(box.conf[0].cpu().numpy())
                    
                    # Convert to YOLO format (normalized)
                    x_center = (x1 + x2) / 2 / w
                    y_center = (y1 + y2) / 2 / h
                    width = (x2 - x1) / w
                    height = (y2 - y1) / h
                    
                    # Save detections with confidence above threshold
                    if conf > 0.3:
                        f.write(f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    def create_yaml_config(self):
        """Create YAML config file"""
        # Extract class names from model
        class_names = self.model.names
        
        # Get input name for dataset naming
        input_name = Path(self.input_path).stem
            
        config = {
            'path': str(self.output_dir.absolute()),
            'train': '.',  # Current directory contains both images and labels
            'val': '.',    # Same directory for validation
            'test': '.',   # Same directory for testing
            'nc': len(class_names),
            'names': list(class_names.values()) if isinstance(class_names, dict) else class_names
        }
        
        yaml_path = self.output_dir / f"{input_name}.yaml"
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        print(f"✅ Config file saved at {yaml_path}")
        
        # Display classes
        print(f"\n📋 Detected classes:")
        for i, name in enumerate(config['names']):
            print(f"   {i}: {name}")
    
    def get_dataset_info(self):
        """Display dataset information"""
        images = len(list(self.output_dir.glob("*.jpg")))
        labels = len(list(self.output_dir.glob("*.txt")))
        
        input_name = Path(self.input_path).stem
        
        print(f"""
📊 Final Dataset Statistics:
{'='*40}
📁 Total Images: {images:4d}
📄 Total Labels: {labels:4d}

💾 Dataset Path: {self.output_dir.absolute()}
🔧 Config File: {self.output_dir / f'{input_name}.yaml'}
        """)
        
        print("✅ Dataset creation completed successfully!")
        print(f"📍 You can find your dataset at: {self.output_dir}")
    
    def cleanup_gui(self):
        """Clean up GUI resources"""
        try:
            if self.root and self.root.winfo_exists():
                self.root.quit()
                self.root.destroy()
        except:
            pass
        
        # Additional cleanup
        try:
            import tkinter as tk
            if tk._default_root:
                tk._default_root.quit()
                tk._default_root.destroy()
                tk._default_root = None
        except:
            pass
    
    def run(self):
        """Run complete process with restart option"""
        while True:
            try:
                # Select paths
                if not self.select_paths_gui():
                    print("❌ Process cancelled")
                    self.cleanup_gui()
                    break
                
                # Create directory structure
                self.setup_directories()
                
                # Process input
                if self.input_type == "video":
                    # Get frame interval from user
                    frame_interval = tk.simpledialog.askinteger(
                        "Video Settings",
                        "Enter frame interval:\n(Example: 30 = every 30th frame)",
                        initialvalue=30,
                        minvalue=1,
                        maxvalue=1000
                    )
                    
                    if frame_interval is None:
                        frame_interval = 30
                    
                    self.process_video(frame_interval)
                else:
                    self.process_image_folder()
                
                # Create config file
                self.create_yaml_config()
                
                # Show statistics
                self.get_dataset_info()
                
                print("🎉 Dataset ready for training!")
                
                # Ask if user wants to continue with another dataset
                continue_choice = messagebox.askyesno(
                    "Continue?", 
                    "Dataset created successfully!\n\nDo you want to create another dataset?\n\nYes = Start again\nNo = Exit"
                )
                
                if not continue_choice:
                    print("👋 Exiting program...")
                    self.cleanup_gui()
                    break
                else:
                    print("\n🔄 Starting new dataset creation...\n")
                    # Reset variables for next iteration
                    self.input_path = None
                    self.output_dir = None
                    self.input_type = None
                    # Keep model loaded for efficiency
                
            except Exception as e:
                error_msg = f"Process error: {str(e)}"
                print(f"❌ {error_msg}")
                try:
                    retry = messagebox.askyesnocancel(
                        "Error", 
                        f"{error_msg}\n\nDo you want to try again?\n\nYes = Restart\nNo = Exit\nCancel = Try again"
                    )
                    
                    if retry is None:  # Cancel - try again
                        continue
                    elif retry:  # Yes - restart
                        print("🔄 Restarting...")
                        continue
                    else:  # No - exit
                        break
                except:
                    break
            finally:
                # Cleanup GUI for this iteration
                try:
                    if self.root and self.root.winfo_exists():
                        self.root.quit()
                        self.root.destroy()
                        self.root = None
                except:
                    pass
        
        # Final cleanup
        self.cleanup_gui()
        print("🔚 Program terminated.")

# Run the program
if __name__ == "__main__":
    try:
        generator = YOLODatasetGenerator()
        generator.run()
        
    except ImportError as e:
        print("❌ Error: tkinter library not installed")
        print("To install: pip install tk")
        input("Press Enter to exit...")
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        input("Press Enter to exit...")
    finally:
        try:
            # Force close any remaining tkinter windows
            import tkinter as tk
            root = tk._default_root
            if root:
                root.quit()
                root.destroy()
        except:
            pass
        
        print("\n🔚 Program finished. Goodbye! 👋")