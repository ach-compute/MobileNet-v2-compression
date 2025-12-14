import csv
import wandb
import os

# Settings
CSV_FILE = "final_benchmark.csv"
PROJECT_NAME = "mobilenet-quantization-assignment"
ORIGINAL_SIZE_MB = 9.0  # Approx size of FP32 MobileNetV2

def upload_to_wandb():
    if not os.path.exists(CSV_FILE):
        print(f"Error: {CSV_FILE} not found. Run run_all_tests.py first.")
        return

    print(f"[*] Reading {CSV_FILE} and uploading to WandB project: '{PROJECT_NAME}'...")

    # Open the CSV file using standard Python library
    with open(CSV_FILE, mode='r') as f:
        # DictReader automatically uses the first row as keys (headers)
        reader = csv.DictReader(f)
        
        count = 0
        for row in reader:
            # The CSV keys will be: 'Model', 'W_Bits', 'A_Bits', 'Size_MB', 'Accuracy'
            # We must convert strings to numbers manually
            try:
                w_bits = int(row['W_Bits'])
                a_bits = int(row['A_Bits'])
                size_mb = float(row['Size_MB'])
                accuracy = float(row['Accuracy'])
            except ValueError:
                continue # Skip bad rows if any

            # Calculate Compression Ratio
            comp_ratio = ORIGINAL_SIZE_MB / size_mb

            # 1. Initialize WandB Run
            run = wandb.init(
                project=PROJECT_NAME,
                name=f"W{w_bits}_A{a_bits}",
                config={
                    "weight_bits": w_bits,
                    "activation_bits": a_bits,
                    "architecture": "MobileNetV2"
                },
                reinit=True
            )

            # 2. Log Metrics
            wandb.log({
                "accuracy": accuracy,
                "model_size_mb": size_mb,
                "compression_ratio": comp_ratio
            })

            # 3. Finish Run
            run.finish()
            print(f"    -> Logged W{w_bits}/A{a_bits} : Acc {accuracy}%")
            count += 1

    print(f"\n[*] Upload Complete. {count} runs logged.")
    print(f"[*] View your Parallel Coordinates Plot here: https://wandb.ai/home")

if __name__ == "__main__":
    upload_to_wandb()
