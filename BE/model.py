import torch
import torch.nn as nn
from torchvision import models, transforms
import torch.nn.functional as F

model_path = 'best_model.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
def load_model(path=model_path):
    model = CNN()
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])
    return model, transform

def predict_image(model, transform, pil_image):
    device = next(model.parameters()).device
    x = transform(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        classes = ['cat', 'dog']
        idx = torch.argmax(probs).item()
        return classes[idx], {classes[i]: float(probs[i]) for i in range(2)}