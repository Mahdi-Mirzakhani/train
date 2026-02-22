import cv2
import sys
import time
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
from queue import Queue, Empty
from collections import OrderedDict

import numpy as np
from ultralytics import YOLO

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QCheckBox, QFileDialog, QMessageBox,
    QFrame, QSizePolicy, QStatusBar, QProgressBar, QGridLayout, QSpinBox,
    QGroupBox, QFormLayout, QComboBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize
from PyQt6.QtGui import QImage, QPixmap, QFont, QPalette, QColor, QIcon


@dataclass
class AppConfig:
    """Application Configuration"""
    model1_path: str = r"yolov11n-face.pt"
    model2_path: str = r"D:\train\new-dataset\image3_dataset\dataset\runs\detect\train2\weights\best.pt"
    video_path: str = r"D:\python\detect-face\video\3.MP4"
    
    frame_width: int = 900
    frame_height: int = 500
    
    # BGR format for OpenCV
    model1_color: Tuple[int, int, int] = (0, 80, 255)      # Dark Orange
    model2_color: Tuple[int, int, int] = (0, 180, 0)       # Dark Green
    box_thickness: int = 3
    
    cache_size: int = 50
    prefetch_frames: int = 10
    
    auto_pause_on_diff: bool = False
    detection_diff_threshold: int = 10  # تفاوت تعداد تشخیص برای توقف خودکار
    show_confidence: bool = True
    confidence_threshold: float = 0.55
    log_level: int = logging.INFO


class FrameCache:
    """Smart Frame Cache with LRU Policy"""
    
    def __init__(self, max_size: int = 50):
        self.cache: OrderedDict[int, Tuple] = OrderedDict()
        self.max_size = max_size
    
    def get(self, frame_num: int) -> Optional[Tuple]:
        if frame_num in self.cache:
            self.cache.move_to_end(frame_num)
            return self.cache[frame_num]
        return None
    
    def put(self, frame_num: int, data: Tuple):
        if frame_num in self.cache:
            self.cache.move_to_end(frame_num)
        else:
            self.cache[frame_num] = data
            if len(self.cache) > self.max_size:
                self.cache.popitem(last=False)
    
    def clear(self):
        self.cache.clear()


class ProcessingThread(QThread):
    """Video Processing Thread"""
    
    frame_ready = pyqtSignal(np.ndarray, np.ndarray, dict)
    status_update = pyqtSignal(str)
    progress_update = pyqtSignal(int, int)
    finished_signal = pyqtSignal()
    
    def __init__(self, comparator):
        super().__init__()
        self.comparator = comparator
        self.running = True
        
    def run(self):
        """Execute video processing"""
        try:
            comp = self.comparator
            
            if not comp.load_models():
                self.status_update.emit("Error: Failed to load models")
                return
                
            if not comp.open_video():
                self.status_update.emit("Error: Failed to open video")
                return
            
            self.status_update.emit("Status: Processing started")
            frame_times = []
            
            while self.running and comp.cap.isOpened():
                if comp.paused and not comp.seeking:
                    time.sleep(0.05)
                    continue
                
                if comp.seeking:
                    comp.seek_to_frame(comp.target_frame)
                    comp.seeking = False
                    comp.paused = True
                    continue
                
                frame_start = time.time()
                
                ret, frame = comp.cap.read()
                if not ret:
                    self.status_update.emit("Status: End of video")
                    comp.paused = True
                    time.sleep(0.1)
                    continue
                
                comp.current_frame = int(comp.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                
                try:
                    frame1, frame2, stats = comp.process_frame(frame, comp.current_frame)
                    
                    # Check detection difference threshold
                    detection_diff = abs(stats['model2_detections'] - stats['model1_detections'])
                    
                    if comp.config.auto_pause_on_diff and detection_diff >= comp.config.detection_diff_threshold:
                        comp.diff_count += 1
                        comp.paused = True
                        self.status_update.emit(
                            f"Auto-Pause: Frame {comp.current_frame} | Diff: {detection_diff} detections"
                        )
                    
                    self.frame_ready.emit(frame1, frame2, stats)
                    self.progress_update.emit(comp.current_frame, comp.total_frames)
                    
                except Exception as e:
                    comp.logger.error(f"Frame processing error {comp.current_frame}: {e}")
                
                frame_time = time.time() - frame_start
                frame_times.append(frame_time)
                if len(frame_times) > 30:
                    frame_times.pop(0)
                
                if frame_times:
                    avg_time = sum(frame_times) / len(frame_times)
                    comp.fps_actual = 1.0 / avg_time if avg_time > 0 else 0
                
                target_time = 1.0 / comp.fps_video
                sleep_time = max(0, target_time - frame_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
        except Exception as e:
            self.status_update.emit(f"Error: {str(e)}")
            comp.logger.error(f"Thread error: {e}", exc_info=True)
        finally:
            self.finished_signal.emit()
    
    def stop(self):
        """Stop processing thread"""
        self.running = False


class YOLOComparator:
    """YOLO Models Comparison Engine"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self._setup_logging()
        
        # Control flags
        self.paused = False
        self.stop_flag = False
        self.running = False
        self.show_box_count = True
        self.seeking = False
        self.target_frame = 0
        
        # Resources
        self.cap: Optional[cv2.VideoCapture] = None
        self.model1: Optional[YOLO] = None
        self.model2: Optional[YOLO] = None
        
        # Statistics
        self.current_frame = 0
        self.total_frames = 0
        self.diff_count = 0
        self.fps_video = 30
        self.fps_actual = 0
        self.current_detections = {'model1': 0, 'model2': 0}
        
        # Cache
        self.frame_cache = FrameCache(config.cache_size)
        
    def _setup_logging(self):
        """Setup logging system"""
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
        """Load YOLO models"""
        try:
            self.logger.info("Loading models...")
            
            if not Path(self.config.model1_path).exists():
                raise FileNotFoundError(f"Model 1 not found: {self.config.model1_path}")
            if not Path(self.config.model2_path).exists():
                raise FileNotFoundError(f"Model 2 not found: {self.config.model2_path}")
                
            self.model1 = YOLO(self.config.model1_path)
            self.model2 = YOLO(self.config.model2_path)
            
            # Warm-up
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            self.model1(dummy, conf=self.config.confidence_threshold, verbose=False)
            self.model2(dummy, conf=self.config.confidence_threshold, verbose=False)
            
            self.logger.info("Models loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Model loading error: {e}")
            return False
    
    def open_video(self) -> bool:
        """Open video file"""
        try:
            if not Path(self.config.video_path).exists():
                raise FileNotFoundError(f"Video not found: {self.config.video_path}")
            
            if self.cap:
                self.cap.release()
            
            self.cap = cv2.VideoCapture(self.config.video_path)
            
            if not self.cap.isOpened():
                raise ValueError("Failed to open video")
            
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps_video = self.cap.get(cv2.CAP_PROP_FPS) or 30
            
            self.logger.info(f"Video opened - Frames: {self.total_frames}, FPS: {self.fps_video}")
            return True
            
        except Exception as e:
            self.logger.error(f"Video opening error: {e}")
            return False
    
    def seek_to_frame(self, frame_num: int) -> bool:
        """Seek to specific frame"""
        try:
            if not self.cap or frame_num < 0 or frame_num >= self.total_frames:
                return False
            
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            self.current_frame = frame_num
            self.logger.info(f"Seeked to frame {frame_num}")
            return True
            
        except Exception as e:
            self.logger.error(f"Seek error: {e}")
            return False
    
    def process_frame(self, frame: np.ndarray, frame_num: int) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Process frame with both models"""
        
        # Check cache
        cached = self.frame_cache.get(frame_num)
        if cached:
            return cached
        
        frame1 = frame.copy()
        frame2 = frame.copy()
        
        # Run models
        results1 = self.model1(frame, conf=self.config.confidence_threshold, verbose=False)
        results2 = self.model2(frame, conf=self.config.confidence_threshold, verbose=False)
        
        boxes1 = results1[0].boxes.xyxy.cpu().numpy()
        confs1 = results1[0].boxes.conf.cpu().numpy()
        
        boxes2 = results2[0].boxes.xyxy.cpu().numpy()
        confs2 = results2[0].boxes.conf.cpu().numpy()
        
        # Draw boxes - Model 1
        for box, conf in zip(boxes1, confs1):
            x1, y1, x2, y2 = map(int, box[:4])
            
            cv2.rectangle(frame1, (x1, y1), (x2, y2), 
                         self.config.model1_color, self.config.box_thickness)
            
            if self.config.show_confidence:
                conf_text = f"{conf*100:.1f}%"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.65
                font_thickness = 2
                text_size = cv2.getTextSize(conf_text, font, font_scale, font_thickness)[0]
                
                text_x = x1
                text_y = y1 - 10 if y1 - 10 > 25 else y1 + 25
                
                bg_x1 = text_x - 2
                bg_y1 = text_y - text_size[1] - 6
                bg_x2 = text_x + text_size[0] + 4
                bg_y2 = text_y + 4
                
                cv2.rectangle(frame1, (bg_x1, bg_y1), (bg_x2, bg_y2), 
                            self.config.model1_color, -1)
                
                cv2.putText(frame1, conf_text, (text_x, text_y), 
                           font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
        
        # Draw boxes - Model 2
        for box, conf in zip(boxes2, confs2):
            x1, y1, x2, y2 = map(int, box[:4])
            
            cv2.rectangle(frame2, (x1, y1), (x2, y2), 
                         self.config.model2_color, self.config.box_thickness)
            
            if self.config.show_confidence:
                conf_text = f"{conf*100:.1f}%"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.65
                font_thickness = 2
                text_size = cv2.getTextSize(conf_text, font, font_scale, font_thickness)[0]
                
                text_x = x1
                text_y = y1 - 10 if y1 - 10 > 25 else y1 + 25
                
                bg_x1 = text_x - 2
                bg_y1 = text_y - text_size[1] - 6
                bg_x2 = text_x + text_size[0] + 4
                bg_y2 = text_y + 4
                
                cv2.rectangle(frame2, (bg_x1, bg_y1), (bg_x2, bg_y2), 
                            self.config.model2_color, -1)
                
                cv2.putText(frame2, conf_text, (text_x, text_y), 
                           font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
        
        # Display detection count
        if self.show_box_count:
            cv2.putText(frame1, f"Detections: {len(boxes1)}", (20, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(frame2, f"Detections: {len(boxes2)}", (20, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3, cv2.LINE_AA)
        
        # Display frame number
        cv2.putText(frame1, f"Frame: {frame_num}", (20, frame.shape[0] - 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame2, f"Frame: {frame_num}", (20, frame.shape[0] - 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Calculate average confidence
        avg_conf1 = np.mean(confs1) if len(confs1) > 0 else 0
        avg_conf2 = np.mean(confs2) if len(confs2) > 0 else 0
        
        stats = {
            'model1_detections': len(boxes1),
            'model2_detections': len(boxes2),
            'diff_found': abs(len(boxes2) - len(boxes1)) >= self.config.detection_diff_threshold,
            'detection_diff': abs(len(boxes2) - len(boxes1)),
            'avg_conf1': float(avg_conf1),
            'avg_conf2': float(avg_conf2)
        }
        
        result = (frame1, frame2, stats)
        self.frame_cache.put(frame_num, result)
        
        return result
    
    def cleanup(self):
        """Release resources"""
        self.running = False
        if self.cap:
            self.cap.release()
        self.frame_cache.clear()
        self.logger.info(f"Cleanup - {self.current_frame}/{self.total_frames} frames, {self.diff_count} differences")


class ProfessionalButton(QPushButton):
    """Professional styled button"""
    
    def __init__(self, text: str, color: str, parent=None):
        super().__init__(text, parent)
        self.setMinimumHeight(42)
        self.setMinimumWidth(130)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {color};
                color: white;
                border: 2px solid {color};
                border-radius: 6px;
                font-size: 12px;
                font-weight: 600;
                padding: 10px 18px;
                font-family: 'Segoe UI', Arial;
            }}
            QPushButton:hover {{
                background-color: {self._lighten_color(color)};
                border: 2px solid {self._lighten_color(color)};
            }}
            QPushButton:pressed {{
                background-color: {self._darken_color(color)};
            }}
            QPushButton:disabled {{
                background-color: #7a7a7a;
                color: #c0c0c0;
                border: 2px solid #7a7a7a;
            }}
        """)
    
    def _lighten_color(self, color: str) -> str:
        color_map = {
            "#2c5aa0": "#3d6bb5",
            "#2e7d32": "#388e3c",
            "#d84315": "#e64a19",
            "#f57c00": "#fb8c00",
            "#1976d2": "#1e88e5",
            "#5d4037": "#6d4c41",
            "#c62828": "#d32f2f",
        }
        return color_map.get(color, color)
    
    def _darken_color(self, color: str) -> str:
        color_map = {
            "#2c5aa0": "#1e4078",
            "#2e7d32": "#1b5e20",
            "#d84315": "#bf360c",
            "#f57c00": "#e65100",
            "#1976d2": "#1565c0",
            "#5d4037": "#4e342e",
            "#c62828": "#b71c1c",
        }
        return color_map.get(color, color)


class ComparisonGUI(QMainWindow):
    """Professional YOLO Models Comparison Interface"""
    
    def __init__(self, config: AppConfig):
        super().__init__()
        self.config = config
        self.comparator = YOLOComparator(config)
        
        self.processing_thread: Optional[ProcessingThread] = None
        self.seeking_by_slider = False
        
        self.setWindowTitle("YOLO Models Comparison System - Professional Edition")
        self.setMinimumSize(1900, 1050)
        
        self._setup_professional_theme()
        self._build_ui()
        self._setup_shortcuts()
        
    def _setup_professional_theme(self):
        """Setup professional color theme"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QLabel {
                color: #2c3e50;
                font-family: 'Segoe UI', Arial;
            }
            QFrame {
                background-color: #ffffff;
                border: 1px solid #e0e0e0;
                border-radius: 8px;
            }
            QGroupBox {
                background-color: #ffffff;
                border: 2px solid #bdbdbd;
                border-radius: 8px;
                margin-top: 12px;
                font-weight: 600;
                font-size: 13px;
                color: #37474f;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 5px 10px;
                background-color: #ffffff;
                color: #1976d2;
            }
        """)
        
    def _build_ui(self):
        """Build user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(12)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        # Header
        header_frame = self._create_header()
        main_layout.addWidget(header_frame)
        
        # Main content (Videos + Settings)
        content_layout = QHBoxLayout()
        
        # Videos
        video_layout = self._create_video_displays()
        content_layout.addLayout(video_layout, 75)
        
        # Settings panel
        settings_panel = self._create_settings_panel()
        content_layout.addWidget(settings_panel, 25)
        
        main_layout.addLayout(content_layout)
        
        # Timeline
        timeline_frame = self._create_timeline()
        main_layout.addWidget(timeline_frame)
        
        # Controls
        controls_frame = self._create_controls()
        main_layout.addWidget(controls_frame)
        
        # Status bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.setStyleSheet("""
            QStatusBar {
                background-color: #37474f;
                color: #ffffff;
                padding: 8px;
                font-weight: 500;
                font-size: 11px;
            }
        """)
        self.statusBar.showMessage("Status: Ready")
        
    def _create_header(self) -> QFrame:
        """Create header section"""
        header_frame = QFrame()
        header_frame.setMaximumHeight(120)
        header_frame.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #1565c0, stop:1 #1976d2);
                border: none;
                border-radius: 8px;
            }
        """)
        header_layout = QVBoxLayout(header_frame)
        
        # Title
        title_label = QLabel("YOLO Models Comparison System")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_font = QFont("Segoe UI", 10, QFont.Weight.Bold)
        title_label.setFont(title_font)
        title_label.setStyleSheet("color: #ffffff; padding: 12px;")
        header_layout.addWidget(title_label)
        
        # Subtitle
        subtitle_label = QLabel("Professional Object Detection Analysis Tool")
        subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle_font = QFont("Segoe UI", 11)
        subtitle_label.setFont(subtitle_font)
        subtitle_label.setStyleSheet("color: #e3f2fd; padding: 3px;")
        header_layout.addWidget(subtitle_label)
        
        # Stats
        self.stats_label = QLabel("Frame: 0/0 | FPS: 0.0 | Model 1: 0 | Model 2: 0 | Differences: 0")
        self.stats_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        stats_font = QFont("Segoe UI", 12, QFont.Weight.Bold)
        self.stats_label.setFont(stats_font)
        self.stats_label.setStyleSheet("color: #ffffff; padding: 5px; background-color: rgba(0,0,0,0.2); border-radius: 4px;")
        header_layout.addWidget(self.stats_label)
        
        return header_frame
    
    def _create_video_displays(self) -> QVBoxLayout:
        """Create video display section"""
        video_layout = QVBoxLayout()
        video_layout.setSpacing(15)
        
        displays_layout = QHBoxLayout()
        displays_layout.setSpacing(15)
        
        # Model 1
        left_group = QGroupBox("YOLO-V11n-face model")
        left_layout = QVBoxLayout(left_group)
        
        self.video_label1 = QLabel()
        self.video_label1.setMinimumSize(self.config.frame_width, self.config.frame_height)
        self.video_label1.setStyleSheet("""
            background-color: #263238;
            border: 3px solid #d84315;
            border-radius: 6px;
        """)
        self.video_label1.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label1.setScaledContents(False)
        left_layout.addWidget(self.video_label1)
        
        displays_layout.addWidget(left_group)
        
        # Model 2
        right_group = QGroupBox("مدل بومی")
        right_layout = QVBoxLayout(right_group)
        
        self.video_label2 = QLabel()
        self.video_label2.setMinimumSize(self.config.frame_width, self.config.frame_height)
        self.video_label2.setStyleSheet("""
            background-color: #263238;
            border: 3px solid #2e7d32;
            border-radius: 6px;
        """)
        self.video_label2.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label2.setScaledContents(False)
        right_layout.addWidget(self.video_label2)
        
        displays_layout.addWidget(right_group)
        
        video_layout.addLayout(displays_layout)
        
        return video_layout
    
    def _create_settings_panel(self) -> QGroupBox:
        """Create settings panel"""
        settings_group = QGroupBox("Detection Settings")
        settings_layout = QVBoxLayout(settings_group)
        settings_layout.setSpacing(15)
        
        # Confidence threshold
        conf_group = QGroupBox("Confidence Threshold")
        conf_layout = QFormLayout(conf_group)
        
        self.threshold_spinner = QSpinBox()
        self.threshold_spinner.setRange(10, 95)
        self.threshold_spinner.setValue(int(self.config.confidence_threshold * 100))
        self.threshold_spinner.setSuffix(" %")
        self.threshold_spinner.valueChanged.connect(self.change_threshold)
        self.threshold_spinner.setStyleSheet("""
            QSpinBox {
                background-color: #ffffff;
                color: #2c3e50;
                border: 2px solid #90caf9;
                border-radius: 5px;
                padding: 8px;
                font-size: 13px;
                font-weight: 600;
            }
            QSpinBox:focus {
                border: 2px solid #1976d2;
            }
        """)
        conf_layout.addRow("Threshold:", self.threshold_spinner)
        settings_layout.addWidget(conf_group)
        
        # Detection difference threshold
        diff_group = QGroupBox("Auto-Pause Configuration")
        diff_layout = QFormLayout(diff_group)
        
        self.diff_threshold_spinner = QSpinBox()
        self.diff_threshold_spinner.setRange(1, 50)
        self.diff_threshold_spinner.setValue(self.config.detection_diff_threshold)
        self.diff_threshold_spinner.setSuffix(" detections")
        self.diff_threshold_spinner.valueChanged.connect(self.change_diff_threshold)
        self.diff_threshold_spinner.setStyleSheet("""
            QSpinBox {
                background-color: #ffffff;
                color: #2c3e50;
                border: 2px solid #90caf9;
                border-radius: 5px;
                padding: 8px;
                font-size: 13px;
                font-weight: 600;
            }
            QSpinBox:focus {
                border: 2px solid #1976d2;
            }
        """)
        diff_layout.addRow("Difference Threshold:", self.diff_threshold_spinner)
        
        diff_help = QLabel("System will auto-pause when detection\ncount difference reaches this value")
        diff_help.setStyleSheet("color: #757575; font-size: 10px; padding: 5px;")
        diff_help.setWordWrap(True)
        diff_layout.addRow(diff_help)
        
        settings_layout.addWidget(diff_group)
        
        # Display options
        display_group = QGroupBox("Display Options")
        display_layout = QVBoxLayout(display_group)
        
        self.show_conf_cb = QCheckBox("Show Confidence Scores")
        self.show_conf_cb.setChecked(self.config.show_confidence)
        self.show_conf_cb.toggled.connect(self.toggle_confidence)
        self.show_conf_cb.setStyleSheet("""
            QCheckBox {
                color: #37474f;
                font-size: 12px;
                font-weight: 500;
                padding: 6px;
            }
            QCheckBox::indicator {
                width: 20px;
                height: 20px;
                border: 2px solid #90caf9;
                border-radius: 4px;
            }
            QCheckBox::indicator:checked {
                background-color: #1976d2;
                border: 2px solid #1976d2;
            }
        """)
        display_layout.addWidget(self.show_conf_cb)
        
        self.auto_pause_cb = QCheckBox("Enable Auto-Pause on Difference")
        self.auto_pause_cb.setChecked(self.config.auto_pause_on_diff)
        self.auto_pause_cb.toggled.connect(self.toggle_auto_pause)
        self.auto_pause_cb.setStyleSheet("""
            QCheckBox {
                color: #37474f;
                font-size: 12px;
                font-weight: 500;
                padding: 6px;
            }
            QCheckBox::indicator {
                width: 20px;
                height: 20px;
                border: 2px solid #90caf9;
                border-radius: 4px;
            }
            QCheckBox::indicator:checked {
                background-color: #1976d2;
                border: 2px solid #1976d2;
            }
        """)
        display_layout.addWidget(self.auto_pause_cb)
        
        settings_layout.addWidget(display_group)
        
        # Statistics
        stats_group = QGroupBox("Session Statistics")
        stats_layout = QVBoxLayout(stats_group)
        
        self.session_stats_label = QLabel(
            "Total Frames: 0\n"
            "Processed: 0\n"
            "Auto-Pauses: 0\n"
            "Avg FPS: 0.0"
        )
        self.session_stats_label.setStyleSheet("""
            color: #37474f;
            font-size: 11px;
            font-family: 'Consolas', 'Courier New';
            padding: 10px;
            background-color: #f5f5f5;
            border-radius: 4px;
        """)
        stats_layout.addWidget(self.session_stats_label)
        
        settings_layout.addWidget(stats_group)
        
        settings_layout.addStretch()
        
        return settings_group
    
    def _create_timeline(self) -> QFrame:
        """Create timeline section"""
        timeline_frame = QFrame()
        timeline_frame.setMaximumHeight(110)
        timeline_layout = QVBoxLayout(timeline_frame)
        
        # Time display
        time_layout = QHBoxLayout()
        
        time_icon = QLabel("⏱")
        time_icon.setStyleSheet("font-size: 16px; color: #1976d2;")
        time_layout.addWidget(time_icon)
        
        time_label = QLabel("Timeline:")
        time_label.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        time_label.setStyleSheet("color: #37474f;")
        time_layout.addWidget(time_label)
        
        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        self.time_label.setStyleSheet("color: #1976d2;")
        time_layout.addWidget(self.time_label)
        
        time_layout.addStretch()
        timeline_layout.addLayout(time_layout)
        
        # Seekbar
        self.seekbar = QSlider(Qt.Orientation.Horizontal)
        self.seekbar.setMinimum(0)
        self.seekbar.setMaximum(1000)
        self.seekbar.valueChanged.connect(self.on_seek)
        self.seekbar.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #bdbdbd;
                height: 14px;
                background: #e0e0e0;
                margin: 2px 0;
                border-radius: 7px;
            }
            QSlider::handle:horizontal {
                background: #1976d2;
                border: 2px solid #0d47a1;
                width: 24px;
                margin: -6px 0;
                border-radius: 12px;
            }
            QSlider::handle:horizontal:hover {
                background: #1e88e5;
            }
            QSlider::sub-page:horizontal {
                background: #1976d2;
                border-radius: 7px;
            }
        """)
        timeline_layout.addWidget(self.seekbar)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumHeight(6)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: none;
                background-color: #e0e0e0;
                border-radius: 3px;
            }
            QProgressBar::chunk {
                background-color: #2e7d32;
                border-radius: 3px;
            }
        """)
        timeline_layout.addWidget(self.progress_bar)
        
        return timeline_frame
    
    def _create_controls(self) -> QFrame:
        """Create control buttons"""
        controls_frame = QFrame()
        controls_layout = QVBoxLayout(controls_frame)
        
        # Keyboard shortcuts help
        help_label = QLabel("Keyboard Shortcuts: [Space] Play/Pause | [W] Toggle Count | [C] Toggle Confidence | [←→] Skip ±10 | [↑↓] Skip ±1")
        help_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        help_label.setFont(QFont("Segoe UI", 10))
        help_label.setStyleSheet("color: #757575; padding: 6px; background-color: #fafafa; border-radius: 4px;")
        controls_layout.addWidget(help_label)
        
        # Buttons
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(12)
        buttons_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.btn_select = ProfessionalButton("📁 Select Video", "#2c5aa0")
        self.btn_select.clicked.connect(self.select_video)
        buttons_layout.addWidget(self.btn_select)
        
        self.btn_start = ProfessionalButton("▶ Start", "#2e7d32")
        self.btn_start.clicked.connect(self.start_video)
        buttons_layout.addWidget(self.btn_start)
        
        self.btn_pause = ProfessionalButton("⏸ Pause", "#f57c00")
        self.btn_pause.clicked.connect(self.pause_video)
        buttons_layout.addWidget(self.btn_pause)
        
        self.btn_continue = ProfessionalButton("⏩ Resume", "#1976d2")
        self.btn_continue.clicked.connect(self.continue_video)
        buttons_layout.addWidget(self.btn_continue)
        
        self.btn_start_frame = ProfessionalButton("⏮ First", "#5d4037")
        self.btn_start_frame.clicked.connect(self.jump_to_start)
        buttons_layout.addWidget(self.btn_start_frame)
        
        self.btn_end_frame = ProfessionalButton("⏭ Last", "#5d4037")
        self.btn_end_frame.clicked.connect(self.jump_to_end)
        buttons_layout.addWidget(self.btn_end_frame)
        
        self.btn_reset = ProfessionalButton("🔄 Reset", "#d84315")
        self.btn_reset.clicked.connect(self.reset_video)
        buttons_layout.addWidget(self.btn_reset)
        
        self.btn_exit = ProfessionalButton("❌ Exit", "#c62828")
        self.btn_exit.clicked.connect(self.close)
        buttons_layout.addWidget(self.btn_exit)
        
        controls_layout.addLayout(buttons_layout)
        
        return controls_frame
    
    def _setup_shortcuts(self):
        """Setup keyboard shortcuts"""
        from PyQt6.QtGui import QShortcut, QKeySequence
        
        QShortcut(QKeySequence(Qt.Key.Key_Space), self, self.toggle_pause)
        QShortcut(QKeySequence(Qt.Key.Key_W), self, self.toggle_detection_count)
        QShortcut(QKeySequence(Qt.Key.Key_C), self, self.toggle_confidence)
        QShortcut(QKeySequence(Qt.Key.Key_Right), self, lambda: self.skip_frames(10))
        QShortcut(QKeySequence(Qt.Key.Key_Left), self, lambda: self.skip_frames(-10))
        QShortcut(QKeySequence(Qt.Key.Key_Up), self, lambda: self.skip_frames(1))
        QShortcut(QKeySequence(Qt.Key.Key_Down), self, lambda: self.skip_frames(-1))
    
    def select_video(self):
        """Select video file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video File",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.MP4);;All Files (*.*)"
        )
        
        if file_path:
            if self.processing_thread and self.processing_thread.isRunning():
                self.processing_thread.stop()
                self.processing_thread.wait()
            
            self.config.video_path = file_path
            self.comparator.config.video_path = file_path
            self.comparator.frame_cache.clear()
            
            self.video_label1.clear()
            self.video_label2.clear()
            
            QMessageBox.information(self, "Success", f"Video selected:\n{Path(file_path).name}")
            self.statusBar.showMessage(f"Status: Video loaded - {Path(file_path).name}")
    
    def start_video(self):
        """Start processing"""
        if not self.processing_thread or not self.processing_thread.isRunning():
            if not Path(self.config.video_path).exists():
                QMessageBox.critical(self, "Error", "Video file not found!")
                return
            
            self.comparator.stop_flag = False
            self.comparator.paused = False
            
            self.processing_thread = ProcessingThread(self.comparator)
            self.processing_thread.frame_ready.connect(self.update_frame)
            self.processing_thread.status_update.connect(self.update_status)
            self.processing_thread.progress_update.connect(self.update_progress)
            self.processing_thread.finished_signal.connect(self.on_processing_finished)
            self.processing_thread.start()
            
            self.statusBar.showMessage("Status: Processing started")
            self.comparator.logger.info("Processing started")
    
    def pause_video(self):
        self.comparator.paused = True
        self.statusBar.showMessage("Status: Paused")
    
    def continue_video(self):
        self.comparator.paused = False
        self.statusBar.showMessage("Status: Resumed")
    
    def toggle_pause(self):
        if self.comparator.paused:
            self.continue_video()
        else:
            self.pause_video()
    
    def reset_video(self):
        """Reset video"""
        if self.processing_thread and self.processing_thread.isRunning():
            self.processing_thread.stop()
            self.processing_thread.wait()
        
        self.comparator.cleanup()
        self.comparator.frame_cache.clear()
        
        self.video_label1.clear()
        self.video_label2.clear()
        self.statusBar.showMessage("Status: Reset completed")
        
        QTimer.singleShot(300, self.start_video)
    
    def skip_frames(self, delta: int):
        if not self.comparator.running:
            return
        
        new_frame = self.comparator.current_frame + delta
        new_frame = max(0, min(new_frame, self.comparator.total_frames - 1))
        
        self.comparator.target_frame = new_frame
        self.comparator.seeking = True
    
    def jump_to_start(self):
        if self.comparator.running:
            self.comparator.target_frame = 0
            self.comparator.seeking = True
    
    def jump_to_end(self):
        if self.comparator.running:
            self.comparator.target_frame = max(0, self.comparator.total_frames - 10)
            self.comparator.seeking = True
    
    def on_seek(self, value):
        if self.seeking_by_slider or not self.comparator.running:
            return
        
        frame_num = int(value * self.comparator.total_frames / 1000)
        self.comparator.target_frame = max(0, min(frame_num, self.comparator.total_frames - 1))
        self.comparator.seeking = True
    
    def toggle_detection_count(self):
        self.comparator.show_box_count = not self.comparator.show_box_count
        status = "enabled" if self.comparator.show_box_count else "disabled"
        self.statusBar.showMessage(f"Detection count display: {status}")
    
    def toggle_confidence(self):
        self.config.show_confidence = not self.config.show_confidence
        self.comparator.config.show_confidence = self.config.show_confidence
        self.show_conf_cb.setChecked(self.config.show_confidence)
        self.comparator.frame_cache.clear()
        
        status = "enabled" if self.config.show_confidence else "disabled"
        self.statusBar.showMessage(f"Confidence display: {status}")
    
    def toggle_auto_pause(self):
        self.comparator.config.auto_pause_on_diff = self.auto_pause_cb.isChecked()
        status = "enabled" if self.auto_pause_cb.isChecked() else "disabled"
        self.statusBar.showMessage(f"Auto-pause: {status}")
    
    def change_threshold(self, value):
        self.config.confidence_threshold = value / 100.0
        self.comparator.config.confidence_threshold = self.config.confidence_threshold
        self.comparator.frame_cache.clear()
        self.statusBar.showMessage(f"Confidence threshold changed to: {value}%")
    
    def change_diff_threshold(self, value):
        """Change detection difference threshold"""
        self.config.detection_diff_threshold = value
        self.comparator.config.detection_diff_threshold = value
        self.comparator.frame_cache.clear()
        self.statusBar.showMessage(f"Detection difference threshold set to: {value}")
    
    def update_frame(self, frame1: np.ndarray, frame2: np.ndarray, stats: dict):
        """Update frames"""
        pixmap1 = self.numpy_to_pixmap(frame1)
        pixmap2 = self.numpy_to_pixmap(frame2)
        
        self.video_label1.setPixmap(pixmap1)
        self.video_label2.setPixmap(pixmap2)
        
        avg_conf1 = stats.get('avg_conf1', 0) * 100
        avg_conf2 = stats.get('avg_conf2', 0) * 100
        detection_diff = stats.get('detection_diff', 0)
        
        stats_text = (
            f"Frame: {self.comparator.current_frame}/{self.comparator.total_frames} | "
            f"FPS: {self.comparator.fps_actual:.1f} | "
            f"Model 1: {stats['model1_detections']} (Avg Conf: {avg_conf1:.1f}%) | "
            f"Model 2: {stats['model2_detections']} (Avg Conf: {avg_conf2:.1f}%) | "
            f"Diff: {detection_diff} | Auto-Pauses: {self.comparator.diff_count}"
        )
        self.stats_label.setText(stats_text)
        
        # Update session stats
        session_stats = (
            f"Total Frames: {self.comparator.total_frames}\n"
            f"Processed: {self.comparator.current_frame}\n"
            f"Auto-Pauses: {self.comparator.diff_count}\n"
            f"Avg FPS: {self.comparator.fps_actual:.1f}"
        )
        self.session_stats_label.setText(session_stats)
    
    def update_status(self, message: str):
        self.statusBar.showMessage(message)
    
    def update_progress(self, current: int, total: int):
        if total > 0:
            progress = int((current / total) * 100)
            self.progress_bar.setValue(progress)
            
            self.seeking_by_slider = True
            seekbar_value = int((current / total) * 1000)
            self.seekbar.setValue(seekbar_value)
            self.seeking_by_slider = False
            
            current_sec = current / self.comparator.fps_video
            total_sec = total / self.comparator.fps_video
            time_text = f"{self._format_time(current_sec)} / {self._format_time(total_sec)}"
            self.time_label.setText(time_text)
    
    def on_processing_finished(self):
        self.statusBar.showMessage("Status: Processing completed")
    
    def numpy_to_pixmap(self, frame: np.ndarray) -> QPixmap:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        q_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        
        return pixmap.scaled(
            self.config.frame_width,
            self.config.frame_height,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
    
    def _format_time(self, seconds: float) -> str:
        m, s = divmod(int(seconds), 60)
        return f"{m:02d}:{s:02d}"
    
    def closeEvent(self, event):
        reply = QMessageBox.question(
            self,
            'Confirm Exit',
            'Are you sure you want to exit the application?',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            if self.processing_thread and self.processing_thread.isRunning():
                self.processing_thread.stop()
                self.processing_thread.wait()
            
            self.comparator.cleanup()
            event.accept()
        else:
            event.ignore()


def main():
    """Main entry point"""
    app = QApplication(sys.argv)
    
    # Set application-wide font
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    # Set application style
    app.setStyle('Fusion')
    
    config = AppConfig()
    window = ComparisonGUI(config)
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()