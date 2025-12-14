# common.py
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os

def get_mobilenet_cifar10(pretrained=False):
    """Returns MobileNetV2 configured for CIFAR-10 (10 classes)"""
    model = models.mobilenet_v2(pretrained=pretrained)
    # Modify the classifier for 10 classes
    model.classifier[1] = nn.Linear(model.last_channel, 10)
    return model

def get_dataloader(data_dir, batch_size=64):
    """Returns CIFAR-10 Test Loader"""
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory {data_dir} not found")

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    # Resize to 224 as per your training setup
    test_dataset = datasets.CIFAR10(
        root=data_dir, 
        train=False, 
        download=False,
        transform=transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            normalize,
        ])
    )

    return torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
