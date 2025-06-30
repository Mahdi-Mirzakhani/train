import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import pandas as pd
from pathlib import Path
import json
import yaml

class YAMLBasedYOLOAnalyzer:
    def __init__(self, yaml_path):
        """
        Initialize YOLO Dataset Analyzer using YAML file
        
        Args:
            yaml_path (str): Path to data.yaml file
        """
        self.yaml_path = Path(yaml_path)
        self.yaml_data = {}
        self.dataset_root = None
        self.train_images_path = None
        self.train_labels_path = None
        self.val_images_path = None
        self.val_labels_path = None
        self.test_images_path = None
        self.test_labels_path = None
        self.class_names = {}
        self.issues = []
        self.stats = {}
        
    def load_yaml_config(self):
        """Load and parse YAML configuration file"""
        print("📄 Loading YAML configuration...")
        print(f"   YAML file: {self.yaml_path}")
        
        if not self.yaml_path.exists():
            self.issues.append(f"❌ YAML file not found: {self.yaml_path}")
            return False
        
        try:
            with open(self.yaml_path, 'r', encoding='utf-8') as f:
                self.yaml_data = yaml.safe_load(f)
            
            print("✅ YAML file loaded successfully")
            
            # Extract dataset root (yaml file directory)
            self.dataset_root = self.yaml_path.parent
            print(f"   Dataset root: {self.dataset_root}")
            
            # Extract paths
            self.extract_paths()
            
            # Extract class names
            if 'names' in self.yaml_data:
                if isinstance(self.yaml_data['names'], list):
                    self.class_names = {i: name for i, name in enumerate(self.yaml_data['names'])}
                elif isinstance(self.yaml_data['names'], dict):
                    self.class_names = self.yaml_data['names']
                print(f"   Classes found: {len(self.class_names)}")
                for class_id, class_name in self.class_names.items():
                    print(f"     {class_id}: {class_name}")
            
            return True
            
        except Exception as e:
            self.issues.append(f"❌ Error reading YAML file: {e}")
            print(f"❌ Error reading YAML file: {e}")
            return False
    
    def extract_paths(self):
        """Extract dataset paths from YAML"""
        print("\n🗂️ Extracting dataset paths...")
        
        # Function to resolve path
        def resolve_path(path_str):
            if path_str is None:
                return None
            path = Path(path_str)
            if not path.is_absolute():
                # Relative to yaml file location
                path = self.dataset_root / path
            return path
        
        # Extract train paths
        if 'train' in self.yaml_data:
            train_path = resolve_path(self.yaml_data['train'])
            if train_path:
                # Check if it's images directory or parent directory
                if train_path.name == 'images':
                    self.train_images_path = train_path
                    self.train_labels_path = train_path.parent / 'labels'
                else:
                    self.train_images_path = train_path / 'images'
                    self.train_labels_path = train_path / 'labels'
                print(f"   Train images: {self.train_images_path}")
                print(f"   Train labels: {self.train_labels_path}")
        
        # Extract validation paths
        if 'val' in self.yaml_data:
            val_path = resolve_path(self.yaml_data['val'])
            if val_path:
                if val_path.name == 'images':
                    self.val_images_path = val_path
                    self.val_labels_path = val_path.parent / 'labels'
                else:
                    self.val_images_path = val_path / 'images'
                    self.val_labels_path = val_path / 'labels'
                print(f"   Val images: {self.val_images_path}")
                print(f"   Val labels: {self.val_labels_path}")
        
        # Extract test paths
        if 'test' in self.yaml_data:
            test_path = resolve_path(self.yaml_data['test'])
            if test_path:
                if test_path.name == 'images':
                    self.test_images_path = test_path
                    self.test_labels_path = test_path.parent / 'labels'
                else:
                    self.test_images_path = test_path / 'images'
                    self.test_labels_path = test_path / 'labels'
                print(f"   Test images: {self.test_images_path}")
                print(f"   Test labels: {self.test_labels_path}")
    
    def check_paths_exist(self):
        """Check if extracted paths exist"""
        print("\n🔍 Checking if paths exist...")
        
        all_good = True
        
        # Check train paths
        if self.train_images_path:
            if self.train_images_path.exists():
                print(f"✅ Train images found: {self.train_images_path}")
            else:
                print(f"❌ Train images not found: {self.train_images_path}")
                self.issues.append(f"❌ Train images directory not found")
                all_good = False
        
        if self.train_labels_path:
            if self.train_labels_path.exists():
                print(f"✅ Train labels found: {self.train_labels_path}")
            else:
                print(f"❌ Train labels not found: {self.train_labels_path}")
                self.issues.append(f"❌ Train labels directory not found")
                all_good = False
        
        # Check val paths
        if self.val_images_path:
            if self.val_images_path.exists():
                print(f"✅ Val images found: {self.val_images_path}")
            else:
                print(f"❌ Val images not found: {self.val_images_path}")
                self.issues.append(f"❌ Val images directory not found")
        
        if self.val_labels_path:
            if self.val_labels_path.exists():
                print(f"✅ Val labels found: {self.val_labels_path}")
            else:
                print(f"❌ Val labels not found: {self.val_labels_path}")
                self.issues.append(f"❌ Val labels directory not found")
        
        # Check test paths
        if self.test_images_path:
            if self.test_images_path.exists():
                print(f"✅ Test images found: {self.test_images_path}")
            else:
                print(f"❌ Test images not found: {self.test_images_path}")
        
        if self.test_labels_path:
            if self.test_labels_path.exists():
                print(f"✅ Test labels found: {self.test_labels_path}")
            else:
                print(f"❌ Test labels not found: {self.test_labels_path}")
        
        return all_good
    
    def analyze_split(self, split_name, images_path, labels_path):
        """Analyze a specific split (train/val/test)"""
        if not images_path or not labels_path:
            return None
        
        if not images_path.exists() or not labels_path.exists():
            return None
        
        print(f"\n📊 Analyzing {split_name.upper()} split...")
        
        # Get image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(list(images_path.glob(f'*{ext}')))
            image_files.extend(list(images_path.glob(f'*{ext.upper()}')))
        
        # Get label files
        label_files = list(labels_path.glob('*.txt'))
        
        print(f"   Images: {len(image_files)}")
        print(f"   Labels: {len(label_files)}")
        
        # Check matching
        image_stems = {f.stem for f in image_files}
        label_stems = {f.stem for f in label_files}
        
        missing_labels = image_stems - label_stems
        missing_images = label_stems - image_stems
        
        issues = []
        if missing_labels:
            issues.append(f"{len(missing_labels)} images without labels")
            print(f"   ⚠️ {len(missing_labels)} images without labels")
        
        if missing_images:
            issues.append(f"{len(missing_images)} labels without images")
            print(f"   ⚠️ {len(missing_images)} labels without images")
        
        # Analyze labels
        class_counts = Counter()
        annotation_count = 0
        invalid_annotations = 0
        
        for label_file in label_files[:100]:  # Sample first 100
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) != 5:
                        invalid_annotations += 1
                        continue
                    
                    try:
                        class_id = int(parts[0])
                        coords = [float(x) for x in parts[1:]]
                        
                        if all(0 <= x <= 1 for x in coords):
                            class_counts[class_id] += 1
                            annotation_count += 1
                        else:
                            invalid_annotations += 1
                    except:
                        invalid_annotations += 1
            except:
                continue
        
        if invalid_annotations > 0:
            issues.append(f"{invalid_annotations} invalid annotations")
            print(f"   ⚠️ {invalid_annotations} invalid annotations")
        
        print(f"   Valid annotations: {annotation_count}")
        print(f"   Classes found: {len(class_counts)}")
        
        return {
            'image_count': len(image_files),
            'label_count': len(label_files),
            'annotation_count': annotation_count,
            'class_distribution': dict(class_counts),
            'issues': issues,
            'invalid_annotations': invalid_annotations
        }
    
    def run_complete_analysis(self):
        """Run complete analysis on all splits"""
        print("🚀 Starting YAML-Based YOLO Dataset Analysis...")
        print("="*60)
        
        # Load YAML
        if not self.load_yaml_config():
            print("❌ Cannot proceed without valid YAML file")
            return
        
        # Check paths
        if not self.check_paths_exist():
            print("⚠️ Some paths are missing, but continuing with available data...")
        
        # Analyze each split
        results = {}
        
        if self.train_images_path and self.train_labels_path:
            results['train'] = self.analyze_split('train', self.train_images_path, self.train_labels_path)
        
        if self.val_images_path and self.val_labels_path:
            results['val'] = self.analyze_split('val', self.val_images_path, self.val_labels_path)
        
        if self.test_images_path and self.test_labels_path:
            results['test'] = self.analyze_split('test', self.test_images_path, self.test_labels_path)
        
        # Generate summary report
        self.generate_summary_report(results)
        
        return results
    
    def generate_summary_report(self, results):
        """Generate final summary report"""
        print("\n" + "="*60)
        print("📋 DATASET ANALYSIS SUMMARY")
        print("="*60)
        
        # YAML info
        print(f"\n📄 YAML Configuration:")
        print(f"   File: {self.yaml_path}")
        print(f"   Classes: {len(self.class_names)}")
        
        # Dataset overview
        total_images = 0
        total_labels = 0
        total_annotations = 0
        
        print(f"\n📊 Split Overview:")
        for split_name, data in results.items():
            if data:
                print(f"   {split_name.upper()}:")
                print(f"     Images: {data['image_count']}")
                print(f"     Labels: {data['label_count']}")
                print(f"     Annotations: {data['annotation_count']}")
                
                total_images += data['image_count']
                total_labels += data['label_count']
                total_annotations += data['annotation_count']
        
        print(f"\n📈 Total Statistics:")
        print(f"   Total Images: {total_images}")
        print(f"   Total Labels: {total_labels}")
        print(f"   Total Annotations: {total_annotations}")
        
        # Class distribution
        print(f"\n🏷️ Class Distribution:")
        all_classes = Counter()
        for split_name, data in results.items():
            if data and data['class_distribution']:
                for class_id, count in data['class_distribution'].items():
                    all_classes[class_id] += count
        
        for class_id, count in all_classes.most_common():
            class_name = self.class_names.get(class_id, f'Unknown_{class_id}')
            print(f"   {class_name} (ID: {class_id}): {count}")
        
        # Issues summary
        all_issues = []
        for split_name, data in results.items():
            if data and data['issues']:
                for issue in data['issues']:
                    all_issues.append(f"{split_name}: {issue}")
        
        if self.issues:
            all_issues.extend(self.issues)
        
        print(f"\n⚠️ Issues Found ({len(all_issues)}):")
        if all_issues:
            for issue in all_issues:
                print(f"   • {issue}")
        else:
            print("   ✅ No major issues detected!")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        if total_images < 100:
            print("   • Consider collecting more data (minimum 100+ images per class)")
        
        if any(data and data['invalid_annotations'] > 0 for data in results.values()):
            print("   • Fix invalid annotations before training")
        
        if len(all_classes) < 2:
            print("   • Multi-class detection needs at least 2 classes")
        
        if all_classes and max(all_classes.values()) / min(all_classes.values()) > 10:
            print("   • Consider balancing classes (use data augmentation)")
        
        if not all_issues:
            print("   ✅ Dataset is ready for YOLO training!")
        
        print("\n" + "="*60)


# Quick usage function
def quick_yaml_analysis(yaml_path):
    """Quick analysis function"""
    analyzer = YAMLBasedYOLOAnalyzer(yaml_path)
    return analyzer.run_complete_analysis()


# Usage example
if __name__ == "__main__":
    # Replace with your yaml file path
    yaml_file = input("Enter path to your data.yaml file: ").strip()
    
    if not yaml_file:
        yaml_file = "D:\\python\\dataset\\data.yamlD:\\python\\dataset\\data.yaml"  # Default
    
    quick_yaml_analysis(yaml_file)