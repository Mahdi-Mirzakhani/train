import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import numpy as np
from collections import Counter
import time
from PIL import Image
import random

###########################################
#           CONFIG - HIGH ACCURACY
###########################################
DATA_DIR = "D:\\Python2\\train"
BATCH_SIZE = 12              
IMG_SIZE = 224
NUM_CLASSES = 5
EPOCHS_HEAD = 30
EPOCHS_FINETUNE = 50
LR_HEAD = 3e-5              
LR_FINE = 8e-6              
WEIGHT_DECAY = 5e-4         
DROPOUT = 0.35              
MIXUP_ALPHA = 0.15          
LABEL_SMOOTHING = 0.05      # خیلی کم
MODEL_PATH = "face_resnet18_high_accuracy.pth"

# Focal Loss برای کلاس‌های سخت
USE_FOCAL_LOSS = True
FOCAL_GAMMA = 2.0

###########################################
#      FOCAL LOSS
###########################################
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        self.ce = nn.CrossEntropyLoss(reduction='none', label_smoothing=label_smoothing)
    
    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        return focal_loss.mean()

###########################################
#      AUGMENTATION - قوی‌تر
###########################################
class StrongAugmentation:
    def __init__(self, img_size=224):
        self.base = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.strong = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.4),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.12))
        ])
    
    def __call__(self, img, use_strong=True):
        if use_strong and random.random() > 0.3:
            return self.strong(img)
        return self.base(img)

###########################################
#      DATASET با Augmentation خاص
###########################################
class SmartImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, is_train=True):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.is_train = is_train
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            if self.is_train and hasattr(self.transform, '__call__'):
                img = self.transform(img, use_strong=True)
            else:
                img = self.transform(img, use_strong=False)
        
        return img, label

###########################################
#      MIXUP
###########################################
def mixup_data(x, y, alpha=0.15):
    if alpha > 0 and random.random() > 0.8:  # فقط 20%
        lam = np.random.beta(alpha, alpha)
        lam = max(lam, 1 - lam)  # حداقل 0.5
    else:
        return x, y, y, 1.0
    
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

###########################################
#      EVALUATION
###########################################
def evaluate(model, loader, device, class_names, detailed=False):
    model.eval()
    correct = 0
    total = 0
    num_classes = len(class_names)
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    predictions_all = []
    labels_all = []
    confidences_all = []
    
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)
            confs, predicted = torch.max(probs, 1)
            
            predictions_all.extend(predicted.cpu().numpy())
            labels_all.extend(labels.cpu().numpy())
            confidences_all.extend(confs.cpu().numpy())
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            for i in range(labels.size(0)):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i] == label:
                    class_correct[label] += 1
    
    overall_acc = 100 * correct / total if total > 0 else 0
    
    if detailed:
        print("\n  📊 Per-class accuracy:")
        for i, name in enumerate(class_names):
            if class_total[i] > 0:
                acc = 100 * class_correct[i] / class_total[i]
                bar = "█" * int(acc / 2.5)
                print(f"    {name:20s}: {acc:5.1f}% {bar} ({class_correct[i]}/{class_total[i]})")
        
        # توزیع پیش‌بینی‌ها
        pred_counts = Counter(predictions_all)
        print("\n  🎯 Prediction distribution:")
        for i, name in enumerate(class_names):
            count = pred_counts.get(i, 0)
            pct = 100 * count / len(predictions_all) if predictions_all else 0
            bar = "█" * int(pct / 2)
            print(f"    {name:20s}: {count:5d} ({pct:5.1f}%) {bar}")
        
        # میانگین confidence
        avg_conf = np.mean(confidences_all) if confidences_all else 0
        print(f"\n  📈 Average confidence: {avg_conf:.3f}")
    
    return overall_acc

###########################################
#      MAIN
###########################################
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 Using device: {device}")
    
    print("\n" + "="*70)
    print("📊 LOADING DATA")
    print("="*70)
    
    dataset = datasets.ImageFolder(DATA_DIR)
    class_names = dataset.classes
    image_paths = [item[0] for item in dataset.samples]
    labels = [item[1] for item in dataset.samples]
    
    print(f"\nClasses: {class_names}")
    print(f"Total: {len(image_paths)}")
    
    class_counts = Counter(labels)
    for idx, name in enumerate(class_names):
        print(f"  {name:20s}: {class_counts[idx]:4d}")
    
    ###########################################
    #      SPLIT - 12% validation
    ###########################################
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels,
        test_size=0.12,
        stratify=labels,
        random_state=42
    )
    
    print(f"\n✂️  Split: {len(train_paths)} train / {len(val_paths)} val (12%)")
    
    ###########################################
    #      DATASETS
    ###########################################
    train_transform = StrongAugmentation(IMG_SIZE)
    val_transform = StrongAugmentation(IMG_SIZE)
    
    train_dataset = SmartImageDataset(train_paths, train_labels, train_transform, is_train=True)
    val_dataset = SmartImageDataset(val_paths, val_labels, val_transform, is_train=False)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=0, 
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=0, 
        pin_memory=True
    )
    
    ###########################################
    #      MODEL: ResNet18
    ###########################################
    print("\n" + "="*70)
    print("🧠 MODEL: ResNet18 + Focal Loss")
    print("="*70)
    
    model = models.resnet18(weights="IMAGENET1K_V1")
    
    # Head با Dropout کمتر
    model.fc = nn.Sequential(
        nn.Dropout(p=DROPOUT),
        nn.Linear(model.fc.in_features, 256),  # بزرگتر
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Dropout(p=DROPOUT * 0.5),
        nn.Linear(256, NUM_CLASSES)
    )
    
    model = model.to(device)
    
    print(f"💧 Dropout: {DROPOUT}")
    print(f"⚖️  Weight decay: {WEIGHT_DECAY}")
    print(f"🎨 Mixup: {MIXUP_ALPHA} (20% of batches)")
    print(f"📏 Validation: 12% of data")
    print(f"🔥 Focal Loss: {'Enabled' if USE_FOCAL_LOSS else 'Disabled'}")
    
    ###########################################
    #      LOSS
    ###########################################
    if USE_FOCAL_LOSS:
        criterion = FocalLoss(gamma=FOCAL_GAMMA, label_smoothing=LABEL_SMOOTHING)
        print(f"   Focal gamma: {FOCAL_GAMMA}")
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    
    ###########################################
    #   PHASE 1: Train Head
    ###########################################
    print("\n" + "="*70)
    print("⚡ PHASE 1: Training Head")
    print("="*70)
    
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True
    
    optimizer = optim.AdamW(model.fc.parameters(), lr=LR_HEAD, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    best_val_acc = 0
    patience = 12
    patience_counter = 0
    
    print(f"\nMax {EPOCHS_HEAD} epochs (early stop: {patience})\n")
    
    for epoch in range(EPOCHS_HEAD):
        model.train()
        total_loss = 0
        start = time.time()
        
        for batch_idx, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            
            # Mixup
            imgs, labels_a, labels_b, lam = mixup_data(imgs, labels, MIXUP_ALPHA)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            
            if lam < 1.0:
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            else:
                loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if (batch_idx + 1) % 8 == 0:
                print(f"  [{epoch+1}/{EPOCHS_HEAD}] Step [{batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f}", end='\r')
        
        scheduler.step()
        
        elapsed = time.time() - start
        avg_loss = total_loss / len(train_loader)
        train_acc = evaluate(model, train_loader, device, class_names)
        val_acc = evaluate(model, val_loader, device, class_names)
        gap = train_acc - val_acc
        
        print(f"\n  [{epoch+1}/{EPOCHS_HEAD}] {elapsed:.1f}s | Loss: {avg_loss:.4f} | Train: {train_acc:.1f}% | Val: {val_acc:.1f}% | Gap: {gap:.1f}%", end="")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            print(" ⭐")
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "val_acc": val_acc,
                "classes": class_names
            }, MODEL_PATH.replace(".pth", "_head.pth"))
        else:
            patience_counter += 1
            print(f" ({patience_counter}/{patience})")
            if patience_counter >= patience:
                print("\n⏹️  Early stop")
                break
    
    print(f"\n✅ Phase 1 Best: {best_val_acc:.2f}%")
    
    checkpoint = torch.load(MODEL_PATH.replace(".pth", "_head.pth"))
    model.load_state_dict(checkpoint["model"])
    
    print("\n🔍 Phase 1 Results:")
    evaluate(model, val_loader, device, class_names, detailed=True)
    
    ###########################################
    #   PHASE 2: Fine-Tuning
    ###########################################
    print("\n" + "="*70)
    print("🔥 PHASE 2: Fine-Tuning (layer3 + layer4 + fc)")
    print("="*70)
    
    # Unfreeze layer3, layer4, fc
    for param in model.layer3.parameters():
        param.requires_grad = True
    for param in model.layer4.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True
    
    # Learning rates مختلف
    param_groups = [
        {'params': model.layer3.parameters(), 'lr': LR_FINE * 0.5},
        {'params': model.layer4.parameters(), 'lr': LR_FINE},
        {'params': model.fc.parameters(), 'lr': LR_FINE * 2}
    ]
    
    optimizer = optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=7, min_lr=1e-7)
    
    best_val_acc = checkpoint["val_acc"]
    patience = 18
    patience_counter = 0
    
    print(f"\nMax {EPOCHS_FINETUNE} epochs\n")
    
    for epoch in range(EPOCHS_FINETUNE):
        model.train()
        total_loss = 0
        start = time.time()
        
        for batch_idx, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            
            # Mixup کمتر
            if random.random() < 0.1:  # فقط 10%
                imgs, labels_a, labels_b, lam = mixup_data(imgs, labels, MIXUP_ALPHA * 0.5)
            else:
                lam = 1.0
                labels_a = labels_b = labels
            
            optimizer.zero_grad()
            outputs = model(imgs)
            
            if lam < 1.0:
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            else:
                loss = criterion(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            if (batch_idx + 1) % 8 == 0:
                print(f"  [{epoch+1}/{EPOCHS_FINETUNE}] Step [{batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f}", end='\r')
        
        elapsed = time.time() - start
        avg_loss = total_loss / len(train_loader)
        train_acc = evaluate(model, train_loader, device, class_names)
        val_acc = evaluate(model, val_loader, device, class_names)
        gap = train_acc - val_acc
        
        scheduler.step(val_acc)
        
        print(f"\n  [{epoch+1}/{EPOCHS_FINETUNE}] {elapsed:.1f}s | Loss: {avg_loss:.4f} | Train: {train_acc:.1f}% | Val: {val_acc:.1f}% | Gap: {gap:.1f}%", end="")
        
        if gap > 18:
            print(" ⚠️", end="")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            print(" ⭐")
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "val_acc": val_acc,
                "train_acc": train_acc,
                "classes": class_names
            }, MODEL_PATH)
        else:
            patience_counter += 1
            print(f" ({patience_counter}/{patience})")
            if patience_counter >= patience:
                print("\n⏹️  Early stop")
                break
    
    ###########################################
    #      FINAL RESULTS
    ###########################################
    print("\n" + "="*70)
    print("🎯 FINAL RESULTS")
    print("="*70)
    
    # چک کردن وجود مدل
    import os
    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH)
        model.load_state_dict(checkpoint["model"])
        
        train_acc = checkpoint['train_acc']
        val_acc = checkpoint['val_acc']
        gap = train_acc - val_acc
        
        print(f"\n✅ Best: Epoch {checkpoint['epoch']+1}")
        print(f"   Train: {train_acc:.2f}%")
        print(f"   Val:   {val_acc:.2f}%")
        print(f"   Gap:   {gap:.2f}%")
    else:
        print(f"\n⚠️  Final model not found (training might have stopped early)")
        print(f"   Using last trained weights...")
        train_acc = train_acc  # از آخرین epoch
        val_acc = val_acc
        gap = train_acc - val_acc
        print(f"   Train: {train_acc:.2f}%")
        print(f"   Val:   {val_acc:.2f}%")
        print(f"   Gap:   {gap:.2f}%")
    
    print("\n📊 Final Validation Results:")
    final_val = evaluate(model, val_loader, device, class_names, detailed=True)
    
    print("\n" + "="*70)
    if gap < 10:
        print("🎉 Perfect! دقت عالی!")
    elif gap < 15:
        print("✅ Excellent! باید خیلی خوب کار کنه")
    elif gap < 22:
        print("👍 Good! قابل قبوله")
    else:
        print("⚠️  هنوز کمی overfit هست")
    
    print(f"\n💾 Model saved: {MODEL_PATH}")
    print("="*70)

if __name__ == '__main__':
    main()