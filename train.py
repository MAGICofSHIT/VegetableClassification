import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm

def main():
    # 1. 数据集路径和配置
    data_dir = "./Vegetable Images"  # 替换为实际路径
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "validation")
    test_dir = os.path.join(data_dir, "test")

    # 数据预处理
    transform = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # 数据加载器
    data = {
        'train': datasets.ImageFolder(train_dir, transform=transform['train']),
        'val': datasets.ImageFolder(val_dir, transform=transform['val']),
        'test': datasets.ImageFolder(test_dir, transform=transform['test'])
    }
    data_loaders = {
        'train': DataLoader(data['train'], batch_size=64, shuffle=True, num_workers=4, pin_memory=True),
        'val': DataLoader(data['val'], batch_size=64, shuffle=False, num_workers=4, pin_memory=True),
        'test': DataLoader(data['test'], batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    }

    # 类别数量
    num_classes = len(data['train'].classes)

    # 2. 定义模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False  # 冻结预训练权重

    # 替换最后的全连接层
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    # 3. 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

    # 学习率调度
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # 4. 训练和验证函数
    def train_and_validate(model, criterion, optimizer, scheduler, num_epochs=60):
        best_acc = 0.0
        best_model_wts = model.state_dict()

        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            print("-" * 10)

            # 每个 epoch 包括训练和验证阶段
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in tqdm(data_loaders[phase], desc=f"{phase.capitalize()} Phase", leave=False):
                    inputs, labels = inputs.to(device), labels.to(device)

                    # 前向传播
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        _, preds = torch.max(outputs, 1)

                        # 反向传播与优化
                        if phase == 'train':
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / len(data[phase])
                epoch_acc = running_corrects.double() / len(data[phase])

                print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

                # 深拷贝最佳模型
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = model.state_dict()

        print(f"Best Val Acc: {best_acc:.4f}")
        model.load_state_dict(best_model_wts)
        return model

    # 5. 模型训练
    model = train_and_validate(model, criterion, optimizer, scheduler, num_epochs=60)

    # 6. 测试模型
    def test_model(model):
        model.eval()
        running_corrects = 0

        for inputs, labels in tqdm(data_loaders['test'], desc="Testing Phase"):
            inputs, labels = inputs.to(device), labels.to(device)
            with torch.no_grad():
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)

        test_acc = running_corrects.double() / len(data['test'])
        print(f"Test Accuracy: {test_acc:.4f}")

    # 测试模型
    test_model(model)

    # 保存最佳模型
    torch.save(model.state_dict(), "best_model.pth")
    print("Best model saved as 'best_model.pth'")

if __name__ == "__main__":
    main()
