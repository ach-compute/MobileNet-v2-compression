# mobilenet_v2_training.py
# Pretrained MobileNetV2 fine-tuned on CIFAR-10 (224x224) with WandB logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
from torch.cuda.amp import autocast, GradScaler

# Set seeds for reproducibility across runs
torch.manual_seed(42)
import random; random.seed(42)
import numpy as np; np.random.seed(42)
torch.backends.cudnn.benchmark = True

# Initialize WandB for experiment tracking and parallel coordinates visualization
# Logs metrics (train/test loss/acc, LR) and params (width_mult, dropout, batch_size)
wandb.init(project="mobilenetv2-cifar10", config={
    "model": "MobileNetV2",
    "pretrained": True,
    "width_mult": 1.0,
    "dropout": 0.001,
    "batch_size": 128,
    "lr_frozen": 0.01,
    "lr_unfrozen": 0.001,
    "epochs_frozen": 30,
    "epochs_full": 70,
    "optimizer": "SGD",
    "momentum": 0.9,
    "weight_decay": 4e-5,
    "label_smoothing": 0.1,
    "scheduler": "CosineAnnealingLR"
})

# Prepare CIFAR-10 dataset with proper normalization and augmentation
# Resize to 224x224 to match pretrained MobileNetV2 input (ImageNet standard)
# Apply RandomCrop and HorizontalFlip for data augmentation to improve generalization
# Use ImageNet normalization to align with pretrained weights
transform_train = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomCrop(224, padding=28),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load CIFAR-10 dataset (50k train, 10k test images)
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

# DataLoaders with batch=128, num_workers=8 for A100D-10C (fast data loading)
trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=8, pin_memory=True)
testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=8, pin_memory=True)

print("Transforms: Train - Resize(224), RandomCrop(224, pad=28), RandomHorizontalFlip(0.5), Normalize(ImageNet); Test - Resize(224), Normalize(ImageNet)")

# Configure MobileNetV2 and define training strategy
# Model: torchvision MobileNetV2 pretrained on ImageNet, width_mult=1.0 (full capacity)
# Classifier: Replace with Dropout(0.001) + Linear(1280→10) for CIFAR-10’s 10 classes
# BatchNorm: Uses torchvision defaults (momentum=0.1, eps=1e-5) for stability
# Training: Two-phase fine-tuning:
# - Phase 1: Freeze backbone (30 epochs, LR=0.01) to adapt classifier
# - Phase 2: Unfreeze all (up to 70 epochs, LR=0.001) for full fine-tuning
# Optimizer: SGD (momentum=0.9, weight_decay=4e-5, nesterov=True)
# Scheduler: CosineAnnealingLR (T_max=100, eta_min=1e-5) for smooth LR decay
# Regularization: Label smoothing=0.1 to reduce overconfidence
# Early stopping: Patience=15 epochs, delta=0.1% to prevent overfitting
model = torchvision.models.mobilenet_v2(pretrained=True)
model.classifier = nn.Sequential(
    nn.Dropout(0.001),
    nn.Linear(model.last_channel, 10)
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=4e-5, nesterov=True)
scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)
scaler = GradScaler()  # Enable mixed precision for ~30% speedup

# Early stopping parameters
patience = 15
best_test_acc = 0.0
patience_counter = 0

# Training function
def train_epoch(epoch, freeze=False):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for inputs, targets in trainloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        with autocast():  # Mixed precision to reduce memory and speed up
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    avg_loss = running_loss / total
    acc = 100. * correct / total
    return avg_loss, acc

# Evaluation function with loss and accuracy
def evaluate():
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    avg_loss = running_loss / total
    acc = 100. * correct / total
    return avg_loss, acc

# Training loop with WandB logging and early stopping
EPOCHS_FROZEN = 30
EPOCHS_FULL = 70
history = {'train_loss': [], 'train_acc': [], 'test_acc': []}

print("Training: Phase 1 (Frozen Backbone, up to 30 epochs)...")
for param in model.features.parameters():
    param.requires_grad = False  # Freeze backbone for classifier adaptation
for epoch in range(1, EPOCHS_FROZEN + 1):
    train_loss, train_acc = train_epoch(epoch, freeze=True)
    test_loss, test_acc = evaluate()
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['test_acc'].append(test_acc)
    print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")
    
    # Log metrics to WandB for parallel coordinates
    wandb.log({
        "epoch": epoch,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "lr": optimizer.param_groups[0]['lr'],
        "phase": "frozen"
    })
    
    # Early stopping check
    if test_acc > best_test_acc + 0.1:
        best_test_acc = test_acc
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model_phase1.pth')
    else:
        patience_counter += 1
    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch} (Phase 1)")
        break
    scheduler.step()

print("\nTraining: Phase 2 (Full Fine-Tune, up to 70 epochs)...")
for param in model.features.parameters():
    param.requires_grad = True  # Unfreeze for full fine-tuning
optimizer.param_groups[0]['lr'] = 0.001  # Lower LR for stability
patience_counter = 0
for epoch in range(EPOCHS_FROZEN + 1, EPOCHS_FROZEN + EPOCHS_FULL + 1):
    train_loss, train_acc = train_epoch(epoch, freeze=False)
    test_loss, test_acc = evaluate()
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['test_acc'].append(test_acc)
    if (epoch - EPOCHS_FROZEN) % 5 == 0 or epoch == EPOCHS_FROZEN + EPOCHS_FULL:  # Log every 5 for speed
        print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")
    
    # Log metrics to WandB
    wandb.log({
        "epoch": epoch,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "lr": optimizer.param_groups[0]['lr'],
        "phase": "unfrozen"
    })
    
    # Early stopping check
    if test_acc > best_test_acc + 0.1:
        best_test_acc = test_acc
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model_phase2.pth')
    else:
        patience_counter += 1
    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch} (Phase 2)")
        break
    scheduler.step()

# Report final test top-1 accuracy
final_test_acc = history['test_acc'][-1]
print(f"\nQ1(c) Final Test Top-1 Accuracy: {final_test_acc:.2f}%")
torch.save(model.state_dict(), 'mobilenetv2_cifar10_final.pth')

# Plot and save loss/accuracy curves
epochs_range = range(1, len(history['train_acc']) + 1)
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, history['train_loss'], label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Curves')
plt.subplot(1, 2, 2)
plt.plot(epochs_range, history['train_acc'], label='Train Acc')
plt.plot(epochs_range, history['test_acc'], label='Test Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.title('Accuracy Curves')
plt.savefig('mobilenetv2_cifar10_q1_curves.png')
plt.show()

# Finish WandB run
wandb.finish()
