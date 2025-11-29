# train.py
# Tiny transfer-learning script for CPU.
# Usage:
#  python train.py --data_dir ./data --epochs 3 --batch_size 8 --output_model best_model.pth
# built by @thealokverse
# cool shit

import os
import argparse
import tarfile
import urllib.request
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models

def download_and_extract_flowers(dest):
    url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    tgz_path = dest / "flower_photos.tgz"
    if not tgz_path.exists():
        print("Downloading flower_photos (~80MB)...")
        urllib.request.urlretrieve(url, tgz_path)
    else:
        print("Archive already exists.")
    extracted = dest / "flower_photos"
    if not extracted.exists():
        print("Extracting...")
        with tarfile.open(tgz_path) as tar:
            tar.extractall(path=dest)
    else:
        print("Dataset already extracted.")
    return str(extracted)

def create_dataloaders(data_dir, batch_size=8, img_size=224):
    tfms_train = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    tfms_val = transforms.Compose([
        transforms.Resize(int(img_size*1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    train_ds = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=tfms_train)
    val_ds = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=tfms_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, train_ds.classes

def split_train_val(original_folder, out_folder, val_frac=0.2):
    # original_folder contains subfolders (class names). We'll split into train/val nodes.
    from shutil import copy2
    import random
    orig = Path(original_folder)
    out = Path(out_folder)
    out.mkdir(parents=True, exist_ok=True)
    train_dir = out / 'train'
    val_dir = out / 'val'
    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)
    for cls_dir in orig.iterdir():
        if not cls_dir.is_dir(): continue
        cls = cls_dir.name
        files = list(cls_dir.glob('*'))
        random.shuffle(files)
        split = int(len(files) * (1 - val_frac))
        train_files = files[:split]
        val_files = files[split:]
        (train_dir/cls).mkdir(parents=True, exist_ok=True)
        (val_dir/cls).mkdir(parents=True, exist_ok=True)
        for f in train_files:
            copy2(f, train_dir/cls / f.name)
        for f in val_files:
            copy2(f, val_dir/cls / f.name)
    print(f"Split into train/val at {out}")

def build_model(num_classes, freeze_backbone=True):
    model = models.mobilenet_v2(pretrained=True)
    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False
    # replace classifier
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_features, num_classes)
    )
    return model

def train(args):
    data_dir = args.data_dir
    # download dataset
    extracted = download_and_extract_flowers(data_dir)
    # split into train/val
    prepared = Path(data_dir) / "prepared"
    if not (prepared.exists() and (prepared / 'train').exists()):
        split_train_val(extracted, prepared, val_frac=0.2)
    train_loader, val_loader, classes = create_dataloaders(str(prepared), batch_size=args.batch_size)
    print("Classes:", classes)

    device = torch.device("cpu")
    model = build_model(num_classes=len(classes), freeze_backbone=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} - train"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        epoch_loss = running_loss / (len(train_loader.dataset) or 1)
        # validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        acc = correct / (total or 1)
        print(f"Epoch {epoch+1} Loss: {epoch_loss:.4f}  Val Acc: {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'classes': classes
            }, args.output_model)
            print("Saved best model.")
    print("Training done. Best val acc:", best_acc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output_model", type=str, default="best_model.pth")
    args = parser.parse_args()
    train(args)
