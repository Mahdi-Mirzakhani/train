import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO
from PIL import Image, ImageOps
from collections import deque
import numpy as np
import time

###############################################
# CONFIG
###############################################
MODEL_PATH = "face_classifier.pth"
YOLO_MODEL = "best.pt"
IMG_SIZE = 224
CONF_THRESHOLD = 0.60
YOLO_CONF = 0.45
MARGIN = 20
TEMPORAL_FRAMES = 9
SKIP_FRAMES = 1
VIDEO_PATH = "D:\\Python\\Face\\Film\\10.mp4"

# TTA
USE_TTA = True
TTA_FLIPS = True
TTA_SCALES = [0.95, 1.0, 1.05]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 Using device: {device}")

###############################################
# SMART MODEL LOADER
###############################################
def load_model_smart(model_path, device):
    """
    هوشمندانه مدل را لود می‌کنه - تشخیص خودکار معماری
    """
    print(f"\n📦 Loading checkpoint: {model_path}")
    
    try:
        ckpt = torch.load(model_path, map_location=device)
        class_names = ckpt["classes"]
        num_classes = len(class_names)
        state_dict = ckpt["model"]
        
        print(f"🏷️  Classes: {class_names}")
        print(f"🎯 Val Accuracy: {ckpt.get('val_acc', 0):.2f}%")
        
        # تشخیص معماری از روی state_dict
        model_type = None
        fc_hidden_size = None
        
        # بررسی تعداد layers برای تشخیص ResNet18 vs ResNet34
        if 'layer3.5.conv1.weight' in state_dict:
            model_type = 'resnet34'
            print("🔍 Detected: ResNet34")
        elif 'layer3.1.conv1.weight' in state_dict:
            model_type = 'resnet18'
            print("🔍 Detected: ResNet18")
        else:
            print("⚠️  Unknown architecture, trying ResNet34...")
            model_type = 'resnet34'
        
        # تشخیص hidden size از FC layer
        for key in state_dict.keys():
            if 'fc.1.weight' in key:
                fc_hidden_size = state_dict[key].shape[0]
                print(f"🔍 FC Hidden Size: {fc_hidden_size}")
                break
        
        if fc_hidden_size is None:
            print("⚠️  Could not detect FC size, using 256")
            fc_hidden_size = 256
        
        # تشخیص dropout rate
        dropout_rate = 0.5  # default
        if 'fc.0.p' in ckpt:
            dropout_rate = ckpt['fc.0.p']
        
        # ساخت مدل
        if model_type == 'resnet18':
            model = models.resnet18(weights=None)
        else:
            model = models.resnet34(weights=None)
        
        # ساخت FC layer مطابق با checkpoint
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, fc_hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(fc_hidden_size),
            nn.Dropout(p=dropout_rate),
            nn.Linear(fc_hidden_size, num_classes)
        )
        
        # لود کردن weights
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        
        print(f"✅ Model loaded successfully!")
        print(f"   Architecture: {model_type.upper()}")
        print(f"   FC Hidden: {fc_hidden_size}")
        print(f"   Dropout: {dropout_rate}")
        
        return model, class_names
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("\n💡 Tip: Make sure the model file matches the training code")
        exit(1)

###############################################
# Load Model
###############################################
model, class_names = load_model_smart(MODEL_PATH, device)

###############################################
# Preprocessing
###############################################
transform_base = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def preprocess_face(face_bgr, img_size=IMG_SIZE):
    if face_bgr.size == 0:
        return None
    
    h, w = face_bgr.shape[:2]
    if h < 30 or w < 30:
        return None
    
    # بهبود کنتراست
    face_bgr = cv2.convertScaleAbs(face_bgr, alpha=1.1, beta=10)
    
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(face_rgb)
    pil = ImageOps.fit(pil, (img_size, img_size), Image.BICUBIC)
    
    return pil

def predict_with_tta(face_img):
    if face_img is None:
        return "UNKNOWN", 0.0
    
    predictions = []
    
    # Original
    img_tensor = transform_base(face_img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(img_tensor)
        prob = torch.softmax(out, dim=1)
        predictions.append(prob)
    
    if USE_TTA:
        # Horizontal flip
        if TTA_FLIPS:
            img_flip = transforms.functional.hflip(face_img)
            img_tensor = transform_base(img_flip).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(img_tensor)
                prob = torch.softmax(out, dim=1)
                predictions.append(prob)
        
        # Multi-scale
        for scale in TTA_SCALES:
            if scale == 1.0:
                continue
            
            new_size = int(IMG_SIZE * scale)
            img_scaled = face_img.resize((new_size, new_size), Image.BICUBIC)
            img_scaled = ImageOps.fit(img_scaled, (IMG_SIZE, IMG_SIZE), Image.BICUBIC)
            
            img_tensor = transform_base(img_scaled).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(img_tensor)
                prob = torch.softmax(out, dim=1)
                predictions.append(prob)
    
    # میانگین
    avg_prob = torch.stack(predictions).mean(dim=0)
    conf, pred = torch.max(avg_prob, dim=1)
    conf_val = conf.item()
    pred_idx = pred.item()
    
    if conf_val < CONF_THRESHOLD:
        return "UNKNOWN", conf_val
    
    return class_names[pred_idx], conf_val

def predict_face(face_bgr):
    pil_img = preprocess_face(face_bgr)
    return predict_with_tta(pil_img)

###############################################
# Face Tracker
###############################################
class FaceTracker:
    def __init__(self, track_id):
        self.id = track_id
        self.predictions = deque(maxlen=TEMPORAL_FRAMES)
        self.confidences = deque(maxlen=TEMPORAL_FRAMES)
        self.last_seen = 0
        self.bbox = None
        self.bbox_history = deque(maxlen=5)
    
    def update(self, name, conf, bbox, frame_num):
        self.predictions.append(name)
        self.confidences.append(conf)
        self.last_seen = frame_num
        self.bbox = bbox
        self.bbox_history.append(bbox)
    
    def get_smoothed_bbox(self):
        if not self.bbox_history:
            return self.bbox
        bboxes = np.array(list(self.bbox_history))
        smoothed = bboxes.mean(axis=0).astype(int)
        return tuple(smoothed)
    
    def get_smoothed_prediction(self):
        if not self.predictions:
            return "UNKNOWN", 0.0
        
        weights = np.exp(np.linspace(0, 1, len(self.predictions)))
        unique_names = list(set(self.predictions))
        scores = {}
        
        for name in unique_names:
            score = sum(w for p, w in zip(self.predictions, weights) if p == name)
            scores[name] = score
        
        best_name = max(scores, key=scores.get)
        confs = [c for p, c in zip(self.predictions, self.confidences) if p == best_name]
        avg_conf = np.mean(confs) if confs else 0.0
        
        return best_name, avg_conf
    
    def is_stale(self, current_frame, max_age=25):
        return current_frame - self.last_seen > max_age

###############################################
# IoU
###############################################
def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_, y1_, x2_, y2_ = box2
    
    xi1 = max(x1, x1_)
    yi1 = max(y1, y1_)
    xi2 = min(x2, x2_)
    yi2 = min(y2, y2_)
    
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_ - x1_) * (y2_ - y1_)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0

###############################################
# Load YOLO
###############################################
print(f"\n📦 Loading YOLO: {YOLO_MODEL}")
yolo = YOLO(YOLO_MODEL)

###############################################
# Video Processing
###############################################
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"❌ Cannot open video: {VIDEO_PATH}")
    exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"\n📹 Video Info:")
print(f"   Resolution: {width}x{height}")
print(f"   FPS: {fps:.1f}")
print(f"   Total frames: {total_frames}")
print(f"   Duration: {total_frames/fps:.1f}s")

print(f"\n⚙️  Settings:")
print(f"   Confidence: {CONF_THRESHOLD}")
print(f"   Temporal smoothing: {TEMPORAL_FRAMES}")
print(f"   TTA: {'Enabled' if USE_TTA else 'Disabled'}")

SAVE_VIDEO = True
if SAVE_VIDEO:
    output_path = "output_face_recognition.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    print(f"   Output: {output_path}")

print(f"\n▶️  Press 'q' to quit, 's' for screenshot, SPACE to pause\n")

trackers = {}
next_tracker_id = 0
frame_num = 0
last_prediction_frame = -SKIP_FRAMES
paused = False

fps_times = deque(maxlen=30)
last_time = time.time()

total_detections = 0
detection_counts = {name: 0 for name in class_names}
detection_counts["UNKNOWN"] = 0

try:
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_num += 1
            
            current_time = time.time()
            fps_times.append(1.0 / (current_time - last_time + 1e-6))
            last_time = current_time
            current_fps = np.mean(fps_times)
            
            results = yolo(frame, conf=YOLO_CONF, verbose=False)[0]
            
            current_boxes = []
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                
                h, w = frame.shape[:2]
                x1 = max(0, x1 - MARGIN)
                y1 = max(0, y1 - MARGIN)
                x2 = min(w, x2 + MARGIN)
                y2 = min(h, y2 + MARGIN)
                
                current_boxes.append((x1, y1, x2, y2))
            
            matched_trackers = set()
            
            for bbox in current_boxes:
                x1, y1, x2, y2 = bbox
                
                best_tracker = None
                best_iou = 0.25
                
                for tid, tracker in trackers.items():
                    if tracker.bbox is not None:
                        iou = compute_iou(bbox, tracker.bbox)
                        if iou > best_iou:
                            best_iou = iou
                            best_tracker = tid
                
                do_predict = (frame_num - last_prediction_frame) >= SKIP_FRAMES
                
                if best_tracker is not None:
                    if do_predict:
                        face = frame[y1:y2, x1:x2]
                        name, conf = predict_face(face)
                        trackers[best_tracker].update(name, conf, bbox, frame_num)
                        total_detections += 1
                        detection_counts[name] += 1
                    else:
                        trackers[best_tracker].bbox = bbox
                        trackers[best_tracker].last_seen = frame_num
                    
                    matched_trackers.add(best_tracker)
                else:
                    if do_predict:
                        face = frame[y1:y2, x1:x2]
                        name, conf = predict_face(face)
                        new_tracker = FaceTracker(next_tracker_id)
                        new_tracker.update(name, conf, bbox, frame_num)
                        trackers[next_tracker_id] = new_tracker
                        matched_trackers.add(next_tracker_id)
                        next_tracker_id += 1
                        total_detections += 1
                        detection_counts[name] += 1
            
            if do_predict:
                last_prediction_frame = frame_num
            
            stale_ids = [tid for tid, t in trackers.items() if t.is_stale(frame_num)]
            for tid in stale_ids:
                del trackers[tid]
            
            for tid in matched_trackers:
                tracker = trackers[tid]
                if tracker.bbox is None:
                    continue
                
                bbox = tracker.get_smoothed_bbox()
                x1, y1, x2, y2 = bbox
                name, conf = tracker.get_smoothed_prediction()
                
                if name == "UNKNOWN":
                    color = (0, 0, 255)
                elif conf > 0.75:
                    color = (0, 255, 0)
                elif conf > 0.65:
                    color = (0, 200, 255)
                else:
                    color = (0, 165, 255)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                
                label = f"{name} {conf:.2f}"
                font_scale = 0.7
                thickness = 2
                (w_txt, h_txt), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
                )
                
                cv2.rectangle(frame, (x1, y1 - h_txt - 12), 
                             (x1 + w_txt + 10, y1), color, -1)
                
                cv2.putText(frame, label, (x1 + 6, y1 - 7),
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 1)
                cv2.putText(frame, label, (x1 + 5, y1 - 6),
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
            
            overlay = frame.copy()
            cv2.rectangle(overlay, (5, 5), (450, 85), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
            
            info_lines = [
                f"FPS: {current_fps:.1f} | Frame: {frame_num}/{total_frames}",
                f"Faces: {len(matched_trackers)} | Total: {total_detections}",
            ]
            
            y_offset = 25
            for line in info_lines:
                cv2.putText(frame, line, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_offset += 25
            
            progress = frame_num / total_frames
            bar_width = width - 40
            cv2.rectangle(frame, (20, height - 30), (20 + bar_width, height - 20),
                         (50, 50, 50), -1)
            cv2.rectangle(frame, (20, height - 30), 
                         (20 + int(bar_width * progress), height - 20),
                         (0, 255, 0), -1)
            
            if SAVE_VIDEO:
                out_video.write(frame)
        
        cv2.imshow("Face Recognition", frame)
        
        key = cv2.waitKey(1 if not paused else 0) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            screenshot = f"screenshot_{frame_num}.jpg"
            cv2.imwrite(screenshot, frame)
            print(f"📸 Screenshot: {screenshot}")
        elif key == ord(' '):
            paused = not paused
            print(f"{'⏸️  Paused' if paused else '▶️  Resumed'}")

finally:
    cap.release()
    if SAVE_VIDEO:
        out_video.release()
    cv2.destroyAllWindows()
    
    print("\n" + "="*60)
    print("📊 DETECTION STATISTICS")
    print("="*60)
    print(f"\nTotal detections: {total_detections}")
    print("\nDetections per class:")
    for name in class_names + ["UNKNOWN"]:
        count = detection_counts[name]
        if count > 0:
            pct = 100 * count / total_detections if total_detections > 0 else 0
            bar = "█" * int(pct / 2)
            print(f"  {name:20s}: {count:5d} ({pct:5.1f}%) {bar}")
    
    print(f"\n✅ Processing completed!")
    if SAVE_VIDEO:
        print(f"💾 Video saved: {output_path}")