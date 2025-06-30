from ultralytics import YOLO

def train_yolo_model(
    model_name='yolov8n.pt',
    data_yaml='data.yaml',
    epochs=100,
    imgsz=640,
    batch_size=16,
    lr0=0.01,
    project='runs/detect',
    name='train'
):
    print("🚀 شروع آموزش مدل YOLOv8...")

    try:
        model = YOLO(model_name)

        results = model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            lr0=lr0,
            project=project,
            name=name,
            device='cpu',  # 👈 استفاده از CPU
            save=True,
            save_period=10,
            val=True,
            plots=True,
            verbose=True
        )

        print("✅ آموزش کامل شد!")
        return results

    except Exception as e:
        print(f"❌ خطا در آموزش مدل: {str(e)}")
        return None

if __name__ == '__main__':
    results = train_yolo_model()
