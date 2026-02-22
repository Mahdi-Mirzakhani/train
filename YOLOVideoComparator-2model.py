import cv2
import threading
import time
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
from queue import Queue, Empty
from collections import OrderedDict

from ultralytics import YOLO
from tkinter import Tk, Label, Button, Frame, StringVar, messagebox, filedialog, Scale, HORIZONTAL, BooleanVar, Checkbutton
from PIL import Image, ImageTk
import numpy as np


@dataclass
class AppConfig:
    """تنظیمات اپلیکیشن"""
    model1_path: str = r"yolov11n-face.pt"
    model2_path: str = r"D:\\train\\new-dataset\\image3_dataset\dataset\\runs\detect\\train2\weights\\best.pt"
    video_path: str = r"D:\python\detect-face\video\3.MP4"
    
    frame_width: int = 800
    frame_height: int = 450
    
    model1_color: Tuple[int, int, int] = (0, 0, 255)  # قرمز
    model2_color: Tuple[int, int, int] = (0, 255, 0)  # سبز
    box_thickness: int = 2
    
    cache_size: int = 50  # تعداد فریم‌های کش شده
    prefetch_frames: int = 10  # پیش‌بارگذاری
    
    auto_pause_on_diff: bool = False
    log_level: int = logging.INFO


class FrameCache:
    """کش هوشمند برای فریم‌های پردازش شده"""
    
    def __init__(self, max_size: int = 50):
        self.cache: OrderedDict[int, Tuple] = OrderedDict()
        self.max_size = max_size
        self.lock = threading.Lock()
    
    def get(self, frame_num: int) -> Optional[Tuple]:
        with self.lock:
            if frame_num in self.cache:
                # LRU: انتقال به آخر
                self.cache.move_to_end(frame_num)
                return self.cache[frame_num]
        return None
    
    def put(self, frame_num: int, data: Tuple):
        with self.lock:
            if frame_num in self.cache:
                self.cache.move_to_end(frame_num)
            else:
                self.cache[frame_num] = data
                # حذف قدیمی‌ترین
                if len(self.cache) > self.max_size:
                    self.cache.popitem(last=False)
    
    def clear(self):
        with self.lock:
            self.cache.clear()


class YOLOComparator:
    """کلاس اصلی برای مقایسه مدل‌های YOLO"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self._setup_logging()
        
        # کنترل
        self.paused = False
        self.stop_flag = False
        self.running = False
        self.show_box_count = True
        self.seeking = False
        self.target_frame = 0
        
        # منابع
        self.cap: Optional[cv2.VideoCapture] = None
        self.model1: Optional[YOLO] = None
        self.model2: Optional[YOLO] = None
        
        # آمار
        self.current_frame = 0
        self.total_frames = 0
        self.diff_count = 0
        self.fps_video = 30
        self.fps_actual = 0
        self.current_detections = {'model1': 0, 'model2': 0}
        
        # کش
        self.frame_cache = FrameCache(config.cache_size)
        
        # Thread safety
        self.lock = threading.Lock()
        self.display_queue = Queue(maxsize=5)
        
    def _setup_logging(self):
        """راه‌اندازی سیستم لاگ"""
        logging.basicConfig(
            level=self.config.log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('yolo_comparison.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_models(self) -> bool:
        """بارگذاری مدل‌ها"""
        try:
            self.logger.info("در حال بارگذاری مدل‌ها...")
            
            if not Path(self.config.model1_path).exists():
                raise FileNotFoundError(f"مدل 1 پیدا نشد: {self.config.model1_path}")
            if not Path(self.config.model2_path).exists():
                raise FileNotFoundError(f"مدل 2 پیدا نشد: {self.config.model2_path}")
                
            self.model1 = YOLO(self.config.model1_path)
            self.model2 = YOLO(self.config.model2_path)
            
            # بهینه‌سازی: warm-up
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            self.model1(dummy, conf=0.55, verbose=False)
            self.model2(dummy, conf=0.55, verbose=False)
            
            self.logger.info("✓ مدل‌ها با موفقیت بارگذاری شدند")
            return True
            
        except Exception as e:
            self.logger.error(f"خطا در بارگذاری مدل‌ها: {e}")
            return False
    
    def open_video(self) -> bool:
        """باز کردن ویدیو"""
        try:
            if not Path(self.config.video_path).exists():
                raise FileNotFoundError(f"ویدیو پیدا نشد: {self.config.video_path}")
            
            if self.cap:
                self.cap.release()
            
            self.cap = cv2.VideoCapture(self.config.video_path)
            
            if not self.cap.isOpened():
                raise ValueError("خطا در باز کردن ویدیو")
            
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps_video = self.cap.get(cv2.CAP_PROP_FPS) or 30
            
            self.logger.info(f"✓ ویدیو باز شد - فریم‌ها: {self.total_frames}, FPS: {self.fps_video}")
            return True
            
        except Exception as e:
            self.logger.error(f"خطا در باز کردن ویدیو: {e}")
            return False
    
    def seek_to_frame(self, frame_num: int) -> bool:
        """رفتن به فریم خاص"""
        try:
            if not self.cap or frame_num < 0 or frame_num >= self.total_frames:
                return False
            
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            self.current_frame = frame_num
            self.logger.info(f"→ Jump به فریم {frame_num}")
            return True
            
        except Exception as e:
            self.logger.error(f"خطا در seek: {e}")
            return False
    
    def process_frame(self, frame: np.ndarray, frame_num: int) -> Tuple[np.ndarray, np.ndarray, dict]:
        """پردازش یک فریم با هر دو مدل"""
        
        # بررسی کش
        cached = self.frame_cache.get(frame_num)
        if cached:
            return cached
        
        frame1 = frame.copy()
        frame2 = frame.copy()
        
        # اجرای مدل‌ها
        results1 = self.model1(frame, conf=0.55, verbose=False)
        results2 = self.model2(frame, conf=0.55, verbose=False)
        
        boxes1 = results1[0].boxes.xyxy.cpu().numpy()
        boxes2 = results2[0].boxes.xyxy.cpu().numpy()
        
        # رسم باکس‌ها - مدل 1
        for box in boxes1:
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(frame1, (x1, y1), (x2, y2), 
                         self.config.model1_color, self.config.box_thickness)
        
        # رسم باکس‌ها - مدل 2
        for box in boxes2:
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(frame2, (x1, y1), (x2, y2), 
                         self.config.model2_color, self.config.box_thickness)
        
        # نمایش تعداد
        if self.show_box_count:
            cv2.putText(frame1, f"Detections: {len(boxes1)}", (20, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame2, f"Detections: {len(boxes2)}", (20, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        
        # نمایش شماره فریم
        cv2.putText(frame1, f"Frame: {frame_num}", (20, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame2, f"Frame: {frame_num}", (20, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        stats = {
            'model1_detections': len(boxes1),
            'model2_detections': len(boxes2),
            'diff_found': len(boxes2) > len(boxes1)
        }
        
        result = (frame1, frame2, stats)
        
        # ذخیره در کش
        self.frame_cache.put(frame_num, result)
        
        return result
    
    def run(self):
        """اجرای مقایسه"""
        if not self.load_models() or not self.open_video():
            return
        
        self.running = True
        self.diff_count = 0
        start_time = time.time()
        frame_times = []
        
        try:
            while not self.stop_flag and self.cap.isOpened():
                # توقف موقت
                if self.paused and not self.seeking:
                    time.sleep(0.05)
                    continue
                
                # Seek request
                if self.seeking:
                    self.seek_to_frame(self.target_frame)
                    self.seeking = False
                    self.paused = True
                    continue
                
                frame_start = time.time()
                
                # خواندن فریم
                ret, frame = self.cap.read()
                if not ret:
                    self.logger.info("پایان ویدیو")
                    self.paused = True
                    time.sleep(0.1)
                    continue
                
                self.current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                
                # پردازش
                try:
                    frame1, frame2, stats = self.process_frame(frame, self.current_frame)
                    
                    # بررسی تفاوت
                    if stats['diff_found']:
                        self.diff_count += 1
                        if self.config.auto_pause_on_diff:
                            self.paused = True
                            self.logger.info(
                                f"⏸ توقف خودکار - فریم {self.current_frame}: "
                                f"M1={stats['model1_detections']}, M2={stats['model2_detections']}"
                            )
                    
                    # ارسال به نمایش
                    try:
                        self.display_queue.put_nowait((frame1, frame2, stats))
                    except:
                        pass  # صف پر است
                    
                except Exception as e:
                    self.logger.error(f"خطا در پردازش فریم {self.current_frame}: {e}")
                
                # محاسبه FPS
                frame_time = time.time() - frame_start
                frame_times.append(frame_time)
                if len(frame_times) > 30:
                    frame_times.pop(0)
                
                if frame_times:
                    avg_time = sum(frame_times) / len(frame_times)
                    self.fps_actual = 1.0 / avg_time if avg_time > 0 else 0
                
                # کنترل سرعت
                target_time = 1.0 / self.fps_video
                sleep_time = max(0, target_time - frame_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
        except Exception as e:
            self.logger.error(f"خطا در پردازش: {e}", exc_info=True)
        finally:
            self.cleanup()
    
    def cleanup(self):
        """آزادسازی منابع"""
        self.running = False
        if self.cap:
            self.cap.release()
        self.frame_cache.clear()
        self.logger.info(f"پایان - {self.current_frame}/{self.total_frames} فریم، {self.diff_count} تفاوت")


class ComparisonGUI:
    """رابط گرافیکی بهبود یافته"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.comparator = YOLOComparator(config)
        
        self.root = Tk()
        self.root.title("🎬 YOLO Comparator Pro - با Timeline")
        self.root.state('zoomed')
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.processing_thread: Optional[threading.Thread] = None
        self.update_thread_running = False
        self.seeking_by_slider = False
        
        self.auto_pause_var = BooleanVar(value=self.config.auto_pause_on_diff)
        
        self._build_ui()
        self._setup_keyboard_bindings()
        
    def _build_ui(self):
        """ساخت رابط کاربری"""
        
        # Header
        header_frame = Frame(self.root, bg="#1a1a2e")
        header_frame.pack(fill="x", pady=(0, 5))
        
        Label(header_frame, text="🎬 YOLO Models Comparison - Professional Edition", 
              font=("Arial", 18, "bold"), bg="#1a1a2e", fg="#eee").pack(pady=10)
        
        # Stats
        self.stats_var = StringVar(value="آماده...")
        Label(header_frame, textvariable=self.stats_var, font=("Arial", 11),
              bg="#1a1a2e", fg="#0f3").pack(pady=5)
        
        # Auto pause checkbox
        Checkbutton(header_frame, text="⏸ توقف خودکار", variable=self.auto_pause_var,
                   command=self.toggle_auto_pause, bg="#1a1a2e", fg="white",
                   selectcolor="#16213e", font=("Arial", 10)).pack()
        
        # Videos
        video_frame = Frame(self.root, bg="#0f0f23")
        video_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Model 1
        left_frame = Frame(video_frame, bg="#0f0f23")
        left_frame.pack(side="left", padx=5, expand=True, fill="both")
        Label(left_frame, text="🔴 Model 1", font=("Arial", 14, "bold"),
              fg="#ff4757", bg="#0f0f23").pack(pady=3)
        self.video_label1 = Label(left_frame, bg="black")
        self.video_label1.pack(expand=True, fill="both")
        
        # Model 2
        right_frame = Frame(video_frame, bg="#0f0f23")
        right_frame.pack(side="right", padx=5, expand=True, fill="both")
        Label(right_frame, text="🟢 Model 2", font=("Arial", 14, "bold"),
              fg="#2ed573", bg="#0f0f23").pack(pady=3)
        self.video_label2 = Label(right_frame, bg="black")
        self.video_label2.pack(expand=True, fill="both")
        
        # Timeline
        self._build_timeline()
        
        # Controls
        self._build_controls()
    
    def _build_timeline(self):
        """ساخت Timeline و Seekbar"""
        timeline_frame = Frame(self.root, bg="#16213e")
        timeline_frame.pack(fill="x", padx=20, pady=10)
        
        Label(timeline_frame, text="⏱ Timeline:", bg="#16213e", fg="white",
              font=("Arial", 11, "bold")).pack(side="left", padx=5)
        
        self.timeline_var = StringVar(value="00:00 / 00:00")
        Label(timeline_frame, textvariable=self.timeline_var, bg="#16213e",
              fg="#0ff", font=("Arial", 10, "bold")).pack(side="left", padx=10)
        
        # Seekbar
        self.seekbar = Scale(
            timeline_frame,
            from_=0,
            to=100,
            orient=HORIZONTAL,
            bg="#16213e",
            fg="white",
            highlightthickness=0,
            troughcolor="#1a1a2e",
            activebackground="#0f3",
            command=self.on_seek,
            length=600
        )
        self.seekbar.pack(side="left", fill="x", expand=True, padx=10)
        
    def _build_controls(self):
        """دکمه‌های کنترل"""
        controls_frame = Frame(self.root, bg="#16213e")
        controls_frame.pack(fill="x", pady=10, padx=20)
        
        Label(controls_frame, text="⌨️ Space=توقف | W=شمارش | ←→=10 فریم | ↑↓=1 فریم",
              bg="#16213e", fg="#bbb", font=("Arial", 10)).pack(pady=5)
        
        btn_frame = Frame(controls_frame, bg="#16213e")
        btn_frame.pack()
        
        buttons = [
            ("📁 انتخاب ویدیو", self.select_video, "#9b59b6"),
            ("▶ شروع", self.start_video, "#2ecc71"),
            ("⏸ توقف", self.pause_video, "#f39c12"),
            ("⏩ ادامه", self.continue_video, "#3498db"),
            ("⏮ ابتدا", self.jump_to_start, "#1abc9c"),
            ("⏭ پایان", self.jump_to_end, "#1abc9c"),
            ("🔄 ریست", self.reset_video, "#e67e22"),
            ("❌ خروج", self.on_closing, "#e74c3c"),
        ]
        
        for text, cmd, color in buttons:
            Button(btn_frame, text=text, command=cmd, bg=color, fg="white",
                  font=("Arial", 10, "bold"), width=12, height=1).pack(side="left", padx=3)
        
    def _setup_keyboard_bindings(self):
        """کلیدهای میانبر"""
        self.root.bind('<space>', lambda e: self.pause_video() if not self.comparator.paused else self.continue_video())
        self.root.bind('<w>', lambda e: self.toggle_detection_count())
        self.root.bind('<W>', lambda e: self.toggle_detection_count())
        self.root.bind('<Right>', lambda e: self.skip_frames(10))
        self.root.bind('<Left>', lambda e: self.skip_frames(-10))
        self.root.bind('<Up>', lambda e: self.skip_frames(1))
        self.root.bind('<Down>', lambda e: self.skip_frames(-1))
        
    def on_seek(self, value):
        """هندل کردن حرکت seekbar"""
        if self.seeking_by_slider or not self.comparator.running:
            return
        
        frame_num = int(float(value) * self.comparator.total_frames / 100)
        self.comparator.target_frame = max(0, min(frame_num, self.comparator.total_frames - 1))
        self.comparator.seeking = True
        self.comparator.logger.info(f"→ Seeking به فریم {self.comparator.target_frame}")
        
    def skip_frames(self, delta: int):
        """پرش تعداد مشخصی فریم"""
        if not self.comparator.running:
            return
        
        new_frame = self.comparator.current_frame + delta
        new_frame = max(0, min(new_frame, self.comparator.total_frames - 1))
        
        self.comparator.target_frame = new_frame
        self.comparator.seeking = True
        
    def jump_to_start(self):
        """رفتن به ابتدای ویدیو"""
        if self.comparator.running:
            self.comparator.target_frame = 0
            self.comparator.seeking = True
            
    def jump_to_end(self):
        """رفتن به انتهای ویدیو"""
        if self.comparator.running:
            self.comparator.target_frame = max(0, self.comparator.total_frames - 10)
            self.comparator.seeking = True
    
    def toggle_detection_count(self):
        """تغییر نمایش تعداد تشخیص‌ها"""
        self.comparator.show_box_count = not self.comparator.show_box_count
        status = "✓ فعال" if self.comparator.show_box_count else "✗ غیرفعال"
        self.comparator.logger.info(f"نمایش شمارش: {status}")
        
    def toggle_auto_pause(self):
        """تغییر توقف خودکار"""
        self.comparator.config.auto_pause_on_diff = self.auto_pause_var.get()
        status = "✓ فعال" if self.auto_pause_var.get() else "✗ غیرفعال"
        self.comparator.logger.info(f"توقف خودکار: {status}")
        
    def select_video(self):
        """انتخاب فایل ویدیو"""
        was_running = self.comparator.running
        if was_running:
            self.comparator.stop_flag = True
            self.update_thread_running = False
            if self.processing_thread:
                self.processing_thread.join(timeout=2)
        
        file_path = filedialog.askopenfilename(
            title="انتخاب ویدیو",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.MP4"), ("All files", "*.*")]
        )
        
        if file_path:
            self.config.video_path = file_path
            self.comparator.config.video_path = file_path
            self.comparator.frame_cache.clear()
            
            self.video_label1.config(image='')
            self.video_label2.config(image='')
            self.stats_var.set(f"✓ {Path(file_path).name}")
            
            messagebox.showinfo("موفق", f"ویدیو انتخاب شد:\n{Path(file_path).name}")
    
    def start_video(self):
        """شروع پردازش"""
        if not self.comparator.running:
            if not Path(self.config.video_path).exists():
                messagebox.showerror("خطا", "فایل ویدیو پیدا نشد!")
                return
            
            self.comparator.stop_flag = False
            self.comparator.paused = False
            
            self.processing_thread = threading.Thread(target=self.comparator.run, daemon=True)
            self.processing_thread.start()
            
            self.update_thread_running = True
            self.update_display()
            self.comparator.logger.info("▶ شروع")
    
    def pause_video(self):
        self.comparator.paused = True
        
    def continue_video(self):
        self.comparator.paused = False
        
    def reset_video(self):
        """ریست"""
        self.comparator.stop_flag = True
        self.update_thread_running = False
        
        if self.processing_thread:
            self.processing_thread.join(timeout=2)
        
        self.comparator.cleanup()
        self.comparator.frame_cache.clear()
        
        self.video_label1.config(image='')
        self.video_label2.config(image='')
        self.stats_var.set("ریست شد")
        
        time.sleep(0.3)
        self.start_video()
    
    def update_display(self):
        """به‌روزرسانی نمایش"""
        if not self.update_thread_running:
            return
        
        try:
            frame1, frame2, stats = self.comparator.display_queue.get_nowait()
            
            img1 = self._prepare_image(frame1)
            img2 = self._prepare_image(frame2)
            
            self.video_label1.config(image=img1)
            self.video_label2.config(image=img2)
            self.video_label1.image = img1
            self.video_label2.image = img2
            
            # به‌روزرسانی seekbar
            if self.comparator.total_frames > 0:
                self.seeking_by_slider = True
                progress = (self.comparator.current_frame / self.comparator.total_frames) * 100
                self.seekbar.set(progress)
                self.seeking_by_slider = False
                
                # زمان
                current_sec = self.comparator.current_frame / self.comparator.fps_video
                total_sec = self.comparator.total_frames / self.comparator.fps_video
                self.timeline_var.set(f"{self._format_time(current_sec)} / {self._format_time(total_sec)}")
            
            # آمار
            self.stats_var.set(
                f"فریم: {self.comparator.current_frame}/{self.comparator.total_frames} | "
                f"FPS: {self.comparator.fps_actual:.1f} | "
                f"M1: {stats['model1_detections']} | M2: {stats['model2_detections']} | "
                f"تفاوت: {self.comparator.diff_count}"
            )
            
        except Empty:
            pass
        
        self.root.after(10, self.update_display)
    
    def _format_time(self, seconds: float) -> str:
        """فرمت زمان"""
        m, s = divmod(int(seconds), 60)
        return f"{m:02d}:{s:02d}"
    
    def _prepare_image(self, frame: np.ndarray) -> ImageTk.PhotoImage:
        """آماده‌سازی تصویر برای نمایش"""
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = img.resize((self.config.frame_width, self.config.frame_height), Image.LANCZOS)
        return ImageTk.PhotoImage(img)
    
    def on_closing(self):
        """بستن برنامه"""
        if messagebox.askokcancel("خروج", "آیا مطمئن هستید؟"):
            self.update_thread_running = False
            self.comparator.stop_flag = True
            
            if self.processing_thread:
                self.processing_thread.join(timeout=2)
            
            self.root.destroy()
    
    def run(self):
        """اجرای برنامه"""
        self.root.mainloop()


def main():
    """تابع اصلی"""
    config = AppConfig()
    app = ComparisonGUI(config)
    app.run()


if __name__ == "__main__":
    main()