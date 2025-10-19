import torch
from tqdm import tqdm

def test_model(model, test_loader, device=torch.device("cpu")):
    model.eval()  # chuyển model sang chế độ đánh giá (evaluation)
    correct, total = 0, 0

    with torch.no_grad():  # không cần tính gradient khi test
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            # Nếu là bài toán classification nhiều lớp
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    print(f"✅ Test Accuracy: {acc:.2f}%")
    return acc
