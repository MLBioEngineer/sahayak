"""
Model Training Script for Sleep Apnea Detection.
Trains the 1D-CNN + BiLSTM + Attention architecture using Weighted Cross-Entropy,
AdamW optimizer, and Cosine Annealing scheduler.
Saves the best model weights to `backend/models/sleep_apnea_best_model.pt`.
"""
import os
import sys
import time
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple

from model_architecture import SleepApneaNet
from dataset_loader import EcgApneaDataset, generate_physionet_benchmark_dataset


def train_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients in BiLSTM
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.5)
        optimizer.step()
        
        total_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
        
    return total_loss / total, (correct / total) * 100.0


def evaluate_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, float, float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    tp = fp = tn = fn = 0
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            
            p = preds.cpu().numpy()
            t = targets.cpu().numpy()
            tp += np.sum((p == 1) & (t == 1))
            tn += np.sum((p == 0) & (t == 0))
            fp += np.sum((p == 1) & (t == 0))
            fn += np.sum((p == 0) & (t == 1))
            
    acc = (correct / total) * 100.0
    sensitivity = (tp / (tp + fn + 1e-8)) * 100.0
    specificity = (tn / (tn + fp + 1e-8)) * 100.0
    
    return total_loss / total, acc, sensitivity, specificity


def main():
    parser = argparse.ArgumentParser(description="Train 1D-CNN+BiLSTM Sleep Apnea Net")
    parser.add_argument("--epochs", type=int, default=35, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Peak learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="AdamW L2 regularization")
    parser.add_argument("--output_dir", type=str, default="../backend/models", help="Destination path for best checkpoint")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f"🚀 Sahayak AI • Sleep Apnea Deep Learning Training")
    print(f"🖥️ Compute Hardware : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"📊 Epochs: {args.epochs} | Batch: {args.batch_size} | LR: {args.lr}")
    print(f"=======================================================\n")

    # 1. Prepare Dataset
    print("[1/4] Preparing PhysioNet Apnea-ECG benchmark cohort...")
    data = generate_physionet_benchmark_dataset(num_subjects=35, epochs_per_subject=40, seq_len=1000)
    
    X_train, y_train = data["train"]
    X_val, y_val = data["val"]
    
    print(f"  ✓ Train Set : {len(y_train)} epochs ({np.sum(y_train == 1)} Apnea, {np.sum(y_train == 0)} Normal)")
    print(f"  ✓ Val Set   : {len(y_val)} epochs ({np.sum(y_val == 1)} Apnea, {np.sum(y_val == 0)} Normal)")

    train_dataset = EcgApneaDataset(X_train, y_train)
    val_dataset = EcgApneaDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # 2. Setup Loss with Class Weights (Inverse Frequency)
    num_normal = np.sum(y_train == 0)
    num_apnea = np.sum(y_train == 1)
    class_weights = torch.tensor([1.0, float(num_normal) / float(num_apnea)], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # 3. Instantiate Architecture
    print("\n[2/4] Initializing 1D-CNN + BiLSTM + Attention Architecture...")
    model = SleepApneaNet(in_channels=1, num_classes=2).to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  ✓ Total Trainable Parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    # 4. Training Loop
    print("\n[3/4] Starting Optimization Loop...")
    best_val_acc = 0.0
    best_epoch = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "sensitivity": [], "specificity": []}

    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, "sleep_apnea_best_model.pt")

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        t_loss, t_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        v_loss, v_acc, sens, spec = evaluate_epoch(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(t_loss)
        history["train_acc"].append(t_acc)
        history["val_loss"].append(v_loss)
        history["val_acc"].append(v_acc)
        history["sensitivity"].append(sens)
        history["specificity"].append(spec)

        is_best = v_acc > best_val_acc
        if is_best:
            best_val_acc = v_acc
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": v_acc,
                "sensitivity": sens,
                "specificity": spec,
                "history": history
            }, save_path)
            star = "⭐ (Best Checkpoint Saved)"
        else:
            star = ""

        if epoch % 5 == 0 or epoch == 1 or is_best:
            print(f"Epoch [{epoch:02d}/{args.epochs:02d}] | Train Loss: {t_loss:.4f} Acc: {t_acc:.1f}% | Val Loss: {v_loss:.4f} Acc: {v_acc:.1f}% (Sens: {sens:.1f}%, Spec: {spec:.1f}%) {star}")

    elapsed = time.time() - start_time
    print(f"\n=======================================================")
    print(f"✅ Training Completed in {elapsed:.1f} seconds!")
    print(f"🏆 Peak Validation Accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})")
    print(f"💾 Checkpoint Exported: {os.path.abspath(save_path)}")
    print(f"=======================================================\n")

    # Save training history JSON
    history_path = os.path.join(args.output_dir, "training_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"📈 Metric logs saved to: {history_path}")


if __name__ == "__main__":
    main()
