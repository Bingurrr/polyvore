import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import timm


batch_size = 1
num_epochs = 10
learning_rate = 0.001
num_classes = len(os.listdir('/home/park/capstone/polyvore/clsdataset'))


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 데이터셋 불러오기
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
dataset = torchvision.datasets.ImageFolder(root='/home/park/capstone/polyvore/clsdataset', transform=data_transforms)

# 데이터셋 분할하기
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# 데이터로더 만들기
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class VisionTransformer(nn.Module):
    def __init__(self, num_classes):
        super(VisionTransformer, self).__init__()
        self.model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)

    def forward(self, x):
        x = self.model(x)
        return x


# 모델 정의하기
model = VisionTransformer(num_classes)
model = model.cuda() 

# 손실 함수 정의하기
criterion = nn.CrossEntropyLoss()

# 최적화 알고리즘 정의하기
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# 모델 학습하기
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # 입력 데이터와 라벨을 GPU 디바이스로 옮기기
        images = images.to(device)
        labels = labels.to(device)

        # 모델 출력값 계산하기
        outputs = model(images)

        # 손실 함수 값 계산하기
        loss = criterion(outputs, labels)

        # 역전파 및 최적화 수행하기
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 로그 출력하기
        if (i + 1) % 10 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))



# 모델 평가하기
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))
