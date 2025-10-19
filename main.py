import numpy as np
from PIL import Image
import torch.nn.functional as F
import torch
import torch.nn as nn
from torchvision import models

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import torch.optim as optim
from tqdm import tqdm
from testModel import test_model

IMG_SIZE = 224
BATCH_SIZE = 32

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)  # Lớp convolutional đầu tiên
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)  # Lớp pooling đầu tiên

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)  # Lớp convolutional thứ hai
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)  # Lớp pooling thứ hai

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)  # Lớp convolutional thứ ba
        self.batchnorm3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)  # Lớp pooling thứ ba

        self.fc1 = nn.Linear(128 * 32 * 32, 256)  # Lớp fully connected
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 2)


    def forward(self, x):
        x = self.pool1(F.relu(self.batchnorm1(self.conv1(x))))
        x = self.pool2(F.relu(self.batchnorm2(self.conv2(x))))
        x = self.pool3(F.relu(self.batchnorm3(self.conv3(x))))

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
def get_model(name="efficientnet"):
    if name == "resnet":
        model = models.resnet50(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 1)
    elif name == "efficientnet":
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, 1)
    elif name == "mobilenet":
        model = models.mobilenet_v2(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, 1)
    elif name == "cnn":
        model = CNN()
    else:
        raise ValueError("Model name not recognized.")
    return model

cnn_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_test = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
])

model_name = "efficientnet"

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_data = datasets.ImageFolder(root='data/CatDogDataset/test', transform=transform_test)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

model = get_model(model_name)
# D:\Intern\GodelnOwnTest\DemoModel\BE\weights\mobilenet_weight.pth
checkpoint_path = f"DemoModel\BE\weights\efficientnet_weight.pth"
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.to(device)

acc = test_model(model, test_loader, device=device)

print(f"✅ Test Accuracy: {acc:.2f}%")
