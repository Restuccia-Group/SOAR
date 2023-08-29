import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from Datagen import CustomDataset
import torchvision.transforms as transforms
from torchvision.transforms import RandomApply, RandomRotation, ColorJitter, RandomGrayscale

# Hyper-Parameters
num_classes = 100
num_epochs = 300
batch_size = 64

if torch.cuda.is_available():
    print('GPU available, Will run the algorithm in: ' + torch.cuda.get_device_name() +
          ' with device capability of: ' + str(torch.cuda.get_device_capability()))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class VGG16(nn.Module):
    def __init__(self, num_classes=num_classes):
        super(VGG16, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  # 227
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 113
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # 113
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 56

        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # 56
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),  # 56
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.layer7 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 28
        )

        self.layer8 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),  # 28
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        self.layer9 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),  # 28
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.layer10 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 14
        )

        self.layer11 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),  # 14
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.layer12 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),  # 14
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.layer13 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 7
        )

        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(7 * 7 * 512, 4096),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU()
        )

        self.fc3 = nn.Sequential(
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.layer11(out)
        out = self.layer12(out)
        out = self.layer13(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out


def prepare_datasets_and_loaders(train_transform, val_test_transform):
    trainset = CustomDataset(csv_file='dataset/CIFAR100/train_set.csv',
                             root_dir='dataset/CIFAR100/TRAIN',
                             transform=train_transform)
    valset = CustomDataset(csv_file='dataset/CIFAR100/val_set.csv',
                           root_dir='dataset/CIFAR100/TRAIN',
                           transform=val_test_transform)
    testset = CustomDataset(csv_file='dataset/CIFAR100/test_set.csv',
                            root_dir='dataset/CIFAR100/TRAIN',
                            transform=val_test_transform)

    train_loader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=valset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, test_loader


def train_and_validate(model, train_loader, val_loader, test_loader, criterion, optimizer, num_epochs, patience=30):
    total_step = len(train_loader)
    PATH = "vgg16.pt"

    best_val_loss = float('inf')
    consecutive_increases = 0

    for epoch in range(num_epochs):
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device, dtype=torch.float)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

        model.eval()
        with torch.no_grad():
            val_loss = 0
            correct = 0
            total = 0
            for images, labels in val_loader:
                images = images.to(device, dtype=torch.float)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            avg_val_loss = val_loss / len(val_loader)
            print('Epoch [{}/{}], Validation Accuracy: {:.2f}%, Validation Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, 100 * correct / total, avg_val_loss))

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                consecutive_increases = 0
                torch.save(model.state_dict(), PATH)
            else:
                consecutive_increases += 1
                if consecutive_increases >= patience:
                    print(f'Validation loss has not increased for {patience} consecutive epochs. Stopping early.')
                    return

    print('Training completed.')
    model.load_state_dict(torch.load(PATH))
    model.eval()

    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device, dtype=torch.float)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Test Accuracy: {:.2f}%'.format(100 * correct / total))


def main():
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(224),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.Resize((227, 227)),
        normalize,
    ])

    val_test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((227, 227)),
        normalize,
    ])

    train_loader, val_loader, test_loader = prepare_datasets_and_loaders(train_transform, val_test_transform)

    model = VGG16(num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, weight_decay=0.005, momentum=0.9)

    train_and_validate(model, train_loader, val_loader, test_loader, criterion, optimizer, num_epochs)

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device, dtype=torch.float)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Test Accuracy: {:.2f}%'.format(100 * correct / total))


if __name__ == "__main__":
    main()
