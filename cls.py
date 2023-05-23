import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import torchvision.models as models


"""
0 : acc
1 : bottom
2 : outer
3 : shoes
4 : top
""" 


model_path = "/home/park/capstone/polyvore/saved_models/my_model.pt"
model = models.resnet18(pretrained=True)
num_classes = 5
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, num_classes)

model.load_state_dict(torch.load(model_path))
model.eval()


image_path = "/home/park/capstone/polyvore/data/img/210/1.jpg"
image = Image.open(image_path).convert('RGB')

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

image = transform(image).unsqueeze(0)


with torch.no_grad():
    outputs = model(image)
    _, predicted = torch.max(outputs.data, 1)

print("Predicted class index:", predicted.item())
