import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
if __name__ == '__main__':
    # ==================== 1. 数据准备 ====================
    # 训练集预处理（含数据增强）
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),          # 随机水平翻转
        transforms.RandomCrop(32, padding=4),       # 随机裁剪
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # 测试集预处理（只做归一化，不做增强）
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # 下载并加载 CIFAR-10 数据集（自动下载，约 160MB）
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform_test)

    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

    # 类别名称
    classes = ('airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    # ==================== 2. 构建 ResNet18 ====================
    def get_resnet18(num_classes=10):
        model = torchvision.models.resnet18(pretrained=False)  # 从头训练
        # 修改最后一层全连接，适应 CIFAR-10 的 10 个类别
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_resnet18(num_classes=10).to(device)

    # ==================== 3. 损失函数 & 优化器 ====================
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)  # 学习率下降

    # ==================== 4. TensorBoard 记录器 ====================
    writer = SummaryWriter('runs/resnet18_cifar10')

    # ==================== 5. 训练循环 ====================
    num_epochs = 10   # 可先设为 10 看效果，再增加
    for epoch in range(num_epochs):
        # ---------- 训练阶段 ----------
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total_train += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()

        avg_train_loss = train_loss / len(trainloader)
        train_acc = 100. * correct_train / total_train

        # ---------- 测试阶段 ----------
        model.eval()
        test_loss = 0.0
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total_test += labels.size(0)
                correct_test += predicted.eq(labels).sum().item()

        avg_test_loss = test_loss / len(testloader)
        test_acc = 100. * correct_test / total_test

        scheduler.step()  # 更新学习率

        # 记录到 TensorBoard
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/test', avg_test_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/test', test_acc, epoch)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)

        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}% | Test Loss: {avg_test_loss:.4f}, Test Acc: {test_acc:.2f}%")

    writer.close()
    print("训练完成！")

    # ==================== 6. 保存模型 ====================
    torch.save(model.state_dict(), 'resnet18_cifar10.pth')

    # ==================== 7. 单张图片预测函数 ====================
    def predict_image(image_path, model, class_names, transform_test):
        """输入图片路径，返回预测类别名称"""
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform_test(image).unsqueeze(0)  # 增加 batch 维度

        device = next(model.parameters()).device
        image_tensor = image_tensor.to(device)

        model.eval()
        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted_idx = outputs.max(1)

        return class_names[predicted_idx.item()]

    # 使用示例（训练完后可以调用）：
    # pred = predict_image('test_img.jpg', model, classes, transform_test)
    # print("预测结果:", pred)