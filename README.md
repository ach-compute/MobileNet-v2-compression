# MobileNet-v2 training and compression on CIFAR-10 with W&B

This project focuses on training MobileNet-v2 on CIFAR-10 and then applying model compression techniques to reduce model size while retaining accuracy. Both accuracy and compression effectiveness will be evaluated.

The script is designed to automatically save the best-performing model from each sweep run as a W&B Artifact, ensuring that the best model is versioned and linked directly to the run that produced it.

## Environment Details
- **Python Version**: 3.10.18 (use `python --version` to confirm)
- **Pip Version**: 25.2 (use `pip --version` to confirm)
- **Operating System**: Ubuntu 20.04.6 LTS (or whatever version you're using; check with `lsb_release -a`)
- **Hardware**: NVIDIA GRID A100D-10C GPU with CUDA 12.2; the script falls back to CPU if a CUDA-enabled GPU is unavailable.
- **WandB Configuration: Requires a WandB account. The script logs to project 2-vgg6-cifar10 under entity abhishek-chaudhary91-iit-madras. Update the entity in the script if needed

## Dependencies
Use the provided `requirements.txt` to install exact versions. Key packages include:
- numpy==2.1.2
- Pillow==11.3.0
- torch==2.5.1+cu121
- torchvision==0.20.1+cu121
- tqdm==4.67.1
- wandb==0.22.0


### Seed Configuration

To ensure run-to-run reproducibility, a fixed seed is configured at the beginning of the script. The following seeds are all set to **`42`**:
-   `torch.manual_seed(42)`
-   `torch.cuda.manual_seed(42)`
-   `numpy.random.seed(42)`
-   `random.seed(42)`

Additionally, `torch.backends.cudnn.deterministic` is set to `True`.

## Setup Instructions

Follow these steps to set up the environment and run the code.

## 1. Create and Activate the Virtual Environment

It is highly recommended to use a virtual environment to manage dependencies.

```bash
# Navigate to your project directory
# Create a Python 3 virtual environment named 'ml_env'
python3 -m venv ml_env

# Activate the virtual environment
source ml_env/bin/activate

# Install dependencies with:
pip install -r requirements.txt
```

## Model trained on CIFAR-10 dataset
We provide a trained MobileNet-v2 model trained on CIFAR-10:
- **Model after training**: checkpoints/mobilenet_v2_cifar10.pth

## Compressed models
After applying compression techniques. Inside output_models folder, below models are present:
- mobilenet_w8_a8.pth
- mobilenet_w8_a6.pth
- mobilenet_w8_a4.pth
- mobilenet_w6_a6.pth
- mobilenet_w5_a5.pth
- mobilenet_w4_a8.pth
- mobilenet_w4_a4.pth
- mobilenet_w3_a3.pth
- mobilenet_w2_a2.pth


## How to test the accuracy of any model
- **Usage**:
```bash
# Test the accuracy of uncompressed (full-precision) model
python test_accuracy.py --model checkpoints/mobilenet_v2_cifar10.pth

# Test the accuracy of compressed/quantized models (examples)
python test_accuracy.py --model output_models/mobilenet_w8_a8.pth
python test_accuracy.py --model output_models/mobilenet_w8_a6.pth
python test_accuracy.py --model output_models/mobilenet_w8_a4.pth
python test_accuracy.py --model output_models/mobilenet_w6_a6.pth
python test_accuracy.py --model output_models/mobilenet_w5_a5.pth
python test_accuracy.py --model output_models/mobilenet_w4_a8.pth
python test_accuracy.py --model output_models/mobilenet_w4_a4.pth
python test_accuracy.py --model output_models/mobilenet_w3_a3.pth
python test_accuracy.py --model output_models/mobilenet_w2_a2.pth
