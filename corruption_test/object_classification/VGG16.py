import torch
import torchvision
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
CUDA_VISDIBLE_DEVICES=0
#import Datagen
from Datagen import CustomDataset
from torch.utils.data import DataLoader

#Hyper-Parameters
num_classes=100
num_epochs=100
batch_size=64


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print('GPU available, Will run the algorithm in: ' + torch.cuda.get_device_name() +
    ' with device capability of: ' + str(torch.cuda.get_device_capability()))


normalize=transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((227, 227)),
    normalize,
])
trainset = CustomDataset(csv_file='dataset/CIFAR100/train_set.csv',
                         root_dir='dataset/CIFAR100/TRAIN',
                         transform=transform)

valset = CustomDataset(csv_file='dataset/CIFAR100/val_set.csv',
                       root_dir='dataset/CIFAR100/TRAIN',
                       transform=transform)


#train_set, val_set = torch.utils.data.random_split(dataset, [30000, 10099])

train_loader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle= True)
val_loader = DataLoader(dataset=valset, batch_size= batch_size, shuffle= True)


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


model = VGG16(num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr= 0.005, weight_decay=0.005, momentum=0.9)

# Train the model
total_step = len(train_loader)

PATH = "vgg16.pt"
loss = criterion

# torch.save({
#     'epoch': num_epochs,
#     'model_state_dict' : model.state_dict(),
#     'optimizer_state_dict' : optimizer.state_dict(),
#     'loss' : loss,
#     }, PATH)

torch.save(model, PATH)

for epoch in range(num_epochs):
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        # Move tensors to the configured device
        images = images.to(device, dtype=torch.float)
        # print(images.shape)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
          .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

    # Validation
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in val_loader:
            images = images.to(device, dtype=torch.float)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            del images, labels, outputs

        print('Accuracy of the network on the {} validation images: {} %'.format(5000, 100 * correct / total))

torch.save(model, PATH)
model = torch.load(PATH)


#Prepare test Data

test_normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],)

test_transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((227,227)),
    test_normalize,
])

testset = CustomDataset(csv_file='dataset/CIFAR100/test_set.csv',
                        root_dir='dataset/CIFAR100/TRAIN',
                        transform=transform)

print(len(testset))


test_loader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=True)

# Do the Testing
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device,dtype= torch.float)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        del images, labels, outputs

    print('Accuracy of the network on the {} test images: {} %'.format(10000, 100 * correct / total))
