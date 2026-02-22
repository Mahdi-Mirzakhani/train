import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
import numpy as np
from collections import Counter
import time
from PIL import Image

###########################################
#           CONFIG
###########################################
DATA_DIR = "D:\\Python\\train"
BATCH_SIZE = 32
IMG_SIZE = 384
NUM_CLASSES = 5
EPOCHS_HEAD = 25
EPOCHS_FINETUNE = 40
LR_HEAD = 1e-3
LR_FINE = 5e-5
WEIGHT_DECAY = 1e-3
MODEL_PATH = "face_efficientnetv2_best.pth"

###########################################
#      TRANSFORMS
###########################################
def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.15))
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

###########################################
#      CUSTOM DATASET
###########################################
class ImageFolderSubset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        img = Image.open(img_path).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, label

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
    
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, predicted = torch.max(outputs, 1)
            
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
            else:
                print(f"    {name:20s}: N/A")
    
    return overall_acc

###########################################
#      MAIN
###########################################
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 Using device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    ###########################################
    #      LOAD DATA
    ###########################################
    print("\n" + "="*70)
    print("📊 DATASET ANALYSIS")
    print("="*70)
    
    dataset = datasets.ImageFolder(DATA_DIR)
    class_names = dataset.classes
    
    # استخراج paths و labels
    image_paths = [item[0] for item in dataset.samples]
    labels = [item[1] for item in dataset.samples]
    
    print(f"\nClasses: {class_names}")
    print(f"Total images: {len(image_paths)}")
    
    class_counts = Counter(labels)
    print("\n📸 Images per class:")
    max_count = max(class_counts.values())
    for idx, name in enumerate(class_names):
        count = class_counts[idx]
        bar = "█" * int(30 * count / max_count)
        print(f"  {name:20s}: {count:4d} {bar}")
    
    ###########################################
    #      SPLIT
    ###########################################
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels,
        test_size=0.18,
        stratify=labels,
        random_state=42
    )
    
    print(f"\n✂️  Split: {len(train_paths)} train / {len(val_paths)} val")
    
    print("\nTrain distribution:")
    train_dist = Counter(train_labels)
    for idx, name in enumerate(class_names):
        print(f"  {name}: {train_dist[idx]}")
    
    print("\nValidation distribution:")
    val_dist = Counter(val_labels)
    for idx, name in enumerate(class_names):
        print(f"  {name}: {val_dist[idx]}")
    
    ###########################################
    #      CREATE DATASETS
    ###########################################
    train_transform, val_transform = get_transforms()
    
    train_dataset = ImageFolderSubset(train_paths, train_labels, train_transform)
    val_dataset = ImageFolderSubset(val_paths, val_labels, val_transform)
    
    ###########################################
    #      WEIGHTED SAMPLER
    ###########################################
    class_weights = 1.0 / torch.tensor([class_counts[i] for i in range(NUM_CLASSES)], dtype=torch.float)
    sample_weights = [class_weights[label] for label in train_labels]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_labels) * 2,
        replacement=True
    )
    
    ###########################################
    #      DATALOADERS
    ###########################################
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    ###########################################
    #      MODEL
    ###########################################
    print("\n" + "="*70)
    print("🧠 BUILDING MODEL: EfficientNet-V2-M")
    print("="*70)
    
    model = models.efficientnet_v2_m(weights="IMAGENET1K_V1")
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    )
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n📐 Total parameters: {total_params:,}")
    
    ###########################################
    #      LOSS
    ###########################################
    class_weights_tensor = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.1)
    
    ###########################################
    #   PHASE 1
    ###########################################
    print("\n" + "="*70)
    print("⚡ PHASE 1: Training Head")
    print("="*70)
    
    for param in model.features.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True
    
    optimizer = optim.AdamW(model.classifier.parameters(), lr=LR_HEAD, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR_HEAD, epochs=EPOCHS_HEAD,
        steps_per_epoch=len(train_loader), pct_start=0.3
    )
    
    best_val_acc = 0
    print(f"\nTraining for {EPOCHS_HEAD} epochs...\n")
    
    for epoch in range(EPOCHS_HEAD):
        model.train()
        total_loss = 0
        start = time.time()
        
        for batch_idx, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            
            if (batch_idx + 1) % 5 == 0:
                print(f"  Epoch [{epoch+1}/{EPOCHS_HEAD}] Step [{batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f}", end='\r')
        
        elapsed = time.time() - start
        avg_loss = total_loss / len(train_loader)
        train_acc = evaluate(model, train_loader, device, class_names)
        val_acc = evaluate(model, val_loader, device, class_names)
        
        print(f"\n  Epoch [{epoch+1}/{EPOCHS_HEAD}] {elapsed:.1f}s | Loss: {avg_loss:.4f} | Train: {train_acc:.1f}% | Val: {val_acc:.1f}%", end="")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(" ⭐")
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "val_acc": val_acc,
                "classes": class_names
            }, MODEL_PATH.replace(".pth", "_head.pth"))
        else:
            print()
    
    print(f"\n✅ Phase 1 Best: {best_val_acc:.2f}%")
    
    checkpoint = torch.load(MODEL_PATH.replace(".pth", "_head.pth"))
    model.load_state_dict(checkpoint["model"])
    evaluate(model, val_loader, device, class_names, detailed=True)
    
    ###########################################
    #   PHASE 2
    ###########################################
    print("\n" + "="*70)
    print("🔥 PHASE 2: Fine-Tuning")
    print("="*70)
    
    total_layers = len(list(model.features.children()))
    unfreeze_from = int(total_layers * 0.6)
    
    for idx, child in enumerate(model.features.children()):
        if idx >= unfreeze_from:
            for param in child.parameters():
                param.requires_grad = True
    
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_FINE, weight_decay=WEIGHT_DECAY * 2
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    
    best_val_acc = checkpoint["val_acc"]
    patience, patience_counter = 10, 0
    
    print(f"\nTraining for up to {EPOCHS_FINETUNE} epochs\n")
    
    for epoch in range(EPOCHS_FINETUNE):
        model.train()
        total_loss = 0
        start = time.time()
        
        for batch_idx, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            if (batch_idx + 1) % 5 == 0:
                print(f"  Epoch [{epoch+1}/{EPOCHS_FINETUNE}] Step [{batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f}", end='\r')
        
        scheduler.step()
        
        elapsed = time.time() - start
        avg_loss = total_loss / len(train_loader)
        train_acc = evaluate(model, train_loader, device, class_names)
        val_acc = evaluate(model, val_loader, device, class_names)
        
        print(f"\n  Epoch [{epoch+1}/{EPOCHS_FINETUNE}] {elapsed:.1f}s | Loss: {avg_loss:.4f} | Train: {train_acc:.1f}% | Val: {val_acc:.1f}%", end="")
        
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
                print(f"\n⏹️  Early stop at epoch {epoch+1}")
                break
    
    ###########################################
    #      FINAL
    ###########################################
    print("\n" + "="*70)
    print("🎯 FINAL RESULTS")
    print("="*70)
    
    checkpoint = torch.load(MODEL_PATH)
    model.load_state_dict(checkpoint["model"])
    
    print(f"\n✅ Best: Epoch {checkpoint['epoch']+1}")
    print(f"   Val: {checkpoint['val_acc']:.2f}%")
    print(f"   Train: {checkpoint['train_acc']:.2f}%")
    
    print("\n📊 Validation:")
    evaluate(model, val_loader, device, class_names, detailed=True)
    
    print("\n📊 Training:")
    evaluate(model, train_loader, device, class_names, detailed=True)
    
    print(f"\n💾 Saved: {MODEL_PATH}")
    print("✨ Done!")

if __name__ == '__main__':
    main()