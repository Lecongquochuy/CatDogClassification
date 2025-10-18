import torch
import torch.nn as nn
from torchvision import models, transforms
import torch.nn.functional as F

model_path = 'weights/best_model.pth'
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

def get_model(model_name, weight_path):
    if model_name == "cnn":
        model = CNN()
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        ])
    elif model_name == "resnet":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 1)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    elif model_name == "mobilenet":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, 1)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    elif model_name == "efficientnet":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, 1)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    else:
        raise ValueError("Unsupported model")

    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    model.eval()

    return model, transform

def predict_image(model, transform, pil_image):
    device = next(model.parameters()).device
    x = transform(pil_image).unsqueeze(0).to(device)
    classes = ['cat','dog']

    with torch.no_grad():
        logits = model(x)
        if logits.shape[1] == 1:
            # binary output -> sigmoid
            prob_dog = torch.sigmoid(logits)[0,0].item()
            label = 'dog' if prob_dog > 0.5 else 'cat'
            probs = {'cat': 1-prob_dog, 'dog': prob_dog}
        else:
            # multi-class output -> softmax
            probs_tensor = torch.softmax(logits, dim=1)[0]
            idx = torch.argmax(probs_tensor).item()
            label = classes[idx]
            probs = {classes[i]: float(probs_tensor[i]) for i in range(len(classes))}

    return label, probs