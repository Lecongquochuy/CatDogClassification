import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from testModel import test_model as evaluate
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(model, train_loader, val_loader, epochs=5, lr=1e-4):
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss, running_correct, running_total = 0, 0, 0
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]")

        for imgs, labels in loop:
            imgs, labels = imgs.to(device), labels.unsqueeze(1).float().to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Tính accuracy trên batch hiện tại
            preds = torch.sigmoid(outputs) > 0.5
            running_correct += (preds == labels).sum().item()
            running_total += labels.size(0)
            running_loss += loss.item()

            batch_acc = 100 * running_correct / running_total
            loop.set_postfix(loss=loss.item(), acc=batch_acc)

        # Tính train accuracy toàn epoch
        train_acc = 100 * running_correct / running_total
        train_loss = running_loss / len(train_loader)

        # Tính val accuracy
        val_acc = evaluate(model, val_loader)

        print(f"\nEpoch {epoch+1}/{epochs} ✅ | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%\n")

    return model
