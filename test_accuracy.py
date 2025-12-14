import torch
import argparse
import os
from common import get_dataloader, get_mobilenet_cifar10
from quantizer_lib import load_compressed_model

def validate(model, val_loader, device):
    correct = 0
    total = 0
    model.eval()
    
    print("[*] Running Inference...")
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    acc = 100 * correct / total
    return acc

def load_any_model(model_path, device):
    """
    Smart loader that tries to load as Compressed first, 
    and falls back to Standard FP32 if that fails.
    """
    print(f"[*] Loading model: {model_path}")
    
    try:
        # 1. Try loading as our Custom Compressed Format
        model = load_compressed_model(model_path, device)
        print("    -> Detected Compressed Model format.")
        return model
        
    except (KeyError, RuntimeError):
        # 2. Fallback: It's likely the Original FP32 Model
        print("    -> Not a compressed model. Loading as Standard FP32...")
        
        model = get_mobilenet_cifar10(pretrained=False)
        checkpoint = torch.load(model_path, map_location=device)
        
        # Handle different checkpoint saving styles
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif isinstance(checkpoint, dict):
            state_dict = checkpoint
        else:
            raise ValueError("Unknown checkpoint format")
            
        # Fix "module." keys if trained with DataParallel
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace("module.", "")
            new_state_dict[name] = v
            
        model.load_state_dict(new_state_dict, strict=True)
        model.to(device)
        return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Model (Original or Compressed)")
    parser.add_argument("--model", type=str, required=True, help="Path to .pth model file")
    parser.add_argument("--data", type=str, default="data", help="Path to data folder")
    args = parser.parse_args()
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Data
    val_loader = get_dataloader(args.data)
    
    # 2. Load Model (Smart Load)
    try:
        model = load_any_model(args.model, DEVICE)
    except Exception as e:
        print(f"[!] Critical Error loading model: {e}")
        exit()
        
    # 3. Test
    acc = validate(model, val_loader, DEVICE)
    print(f"\nResult >> Model: {os.path.basename(args.model)} | Accuracy: {acc:.2f}%")
