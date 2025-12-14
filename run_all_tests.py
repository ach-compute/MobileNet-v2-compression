import torch
import os
import csv
import glob
import re
from common import get_dataloader
from quantizer_lib import load_compressed_model

# --- SETTINGS ---
OUTPUT_DIR = "output_models"
DATA_DIR = "data"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_FILE = "final_benchmark.csv"

def validate(model, val_loader):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def get_bits_from_name(filename):
    # Extracts w8_a8 from "mobilenet_w8_a8.pth"
    match = re.search(r'w(\d+)_a(\d+)', filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return 32, 32

def main():
    # 1. Setup Data
    print("[*] Loading Dataset...")
    val_loader = get_dataloader(DATA_DIR, batch_size=128) # Faster batch size

    # 2. Find all models
    model_files = glob.glob(os.path.join(OUTPUT_DIR, "*.pth"))
    model_files.sort()
    
    # 3. Prepare CSV
    with open(RESULTS_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'W_Bits', 'A_Bits', 'Size_MB', 'Accuracy'])
        
        print(f"\n{'Model File':<25} | {'Size':<6} | {'Acc %':<6}")
        print("-" * 45)
        
        for model_path in model_files:
            filename = os.path.basename(model_path)
            w_bit, a_bit = get_bits_from_name(filename)
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            
            try:
                # Load
                model = load_compressed_model(model_path, DEVICE)
                
                # Test
                acc = validate(model, val_loader)
                
                # Log
                print(f"{filename:<25} | {size_mb:.2f} | {acc:.2f}%")
                writer.writerow([filename, w_bit, a_bit, f"{size_mb:.2f}", f"{acc:.2f}"])
                
            except Exception as e:
                print(f"{filename:<25} | ERROR  | {e}")

    print(f"\n[*] Benchmark complete. Results saved to {RESULTS_FILE}")

if __name__ == "__main__":
    main()
