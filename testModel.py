import torch
from tqdm import tqdm

def test_model(model, test_loader, device):
    model.eval()
    correct, total = 0, 0
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.unsqueeze(1).float().to(device)
        outputs = model(imgs)
        preds = torch.sigmoid(outputs) > 0.5
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return 100 * correct / total

def cnn_test(model, test_loader, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    model.eval()

    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")