# run_sweep.py
import torch
import os
import csv
from common import get_mobilenet_cifar10
from quantizer_lib import ManualQuantizer

# --- SETTINGS ---
MODEL_PATH = "checkpoints/best_model_phase2.pth" # Your original model
OUTPUT_DIR = "output_models"
DATA_DIR = "data"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

# Combinations to try (Weight Bits, Activation Bits)
# Expanded list as requested
combinations = [
    (8, 8), (8, 6), (8, 4),
    (6, 6), (5, 5),
    (4, 8), (4, 4), (3, 3), (2, 2)
]

def load_base_model():
    print(f"[*] Loading base model: {MODEL_PATH}")
    model = get_mobilenet_cifar10(pretrained=False)
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    
    # Fix dictionary keys if needed
    state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    new_state = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state, strict=False)
    return model.to(DEVICE)

def main():
    base_model = load_base_model()
    
    # CSV Logger
    csv_file = open(os.path.join(OUTPUT_DIR, 'sweep_results.csv'), 'w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow(['weight_bits', 'act_bits', 'model_filename', 'size_mb'])
    
    print("\n[*] Starting Quantization Sweep...")
    
    for w_bit, a_bit in combinations:
        print(f"\n--- Config: W{w_bit} / A{a_bit} ---")
        
        # 1. Quantize
        quantizer = ManualQuantizer(base_model)
        quantizer.apply_weight_quantization(w_bit)
        
        # Note: We just register hooks to save the config, 
        # actual activation quantization happens during inference
        quantizer.register_activation_hooks(a_bit)
        
        # 2. Save Compressed
        filename = f"mobilenet_w{w_bit}_a{a_bit}.pth"
        save_path = os.path.join(OUTPUT_DIR, filename)
        
        size_mb = quantizer.save_compressed_model(save_path)
        
        writer.writerow([w_bit, a_bit, save_path, f"{size_mb:.2f}"])
        
    csv_file.close()
    print(f"\n[*] Sweep complete. Models saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
