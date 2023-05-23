import random
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import torchvision.models as models

str = "201_1 201_2 201_3 201_4 202_1 202_2 202_3 202_4 202_5 203_1 203_2 203_3 203_4 203_5 204_1 204_2 204_3 204_4 204_5 205_1 205_2 205_3 205_4 205_5 206_1 206_2 206_3 206_4 207_1 207_2 207_3 207_4 208_1 208_2 208_3 208_4 208_5 209_1 209_2 209_3 209_4 210_1 210_2 210_3 210_4 211_1 211_2 211_3 211_4 212_1 212_2 212_3 212_4 212_5 213_1 213_2 213_3 213_4 213_5 214_1 214_2 214_3 214_4 214_5 214_6 215_1 215_2 215_3 215_4 215_5 216_1 216_2 216_3 216_4 216_5 217_1 217_2 217_3 217_4 217_5 217_6 218_1 218_2 218_3 218_4 218_5 218_6 219_1 219_2 219_3 219_4 219_5 220_1 220_2 220_3 220_4 220_5 221_1 221_2 221_3 221_4 221_5 221_6 222_1 222_2 222_3 222_4 222_5 222_6 223_1 223_2 223_3 223_4 223_5 223_6 224_1 224_2 224_3 224_4 224_5 225_1 225_2 225_3 225_4 225_5 225_6 226_1 226_2 226_3 226_4 226_5 227_1 227_2 227_3 227_4 227_5 228_1 228_2 228_3 228_4 228_5 229_1 229_2 229_3 229_4 229_5 230_1 230_2 230_3 230_4 230_5 231_1 231_2 231_3 231_4 231_5 232_1 232_2 232_3 232_4 232_5 233_1 233_2 233_3 233_4 234_1 234_2 234_3 234_4 234_5 235_1 235_2 235_3 235_4 235_5 236_1 236_2 236_3 236_4 236_5 237_1 237_2 237_3 237_4 237_5 238_1 238_2 238_3 238_4 238_5 239_1 239_2 239_3 239_4 239_5 240_1 240_2 240_3 240_4 240_5"
candidate = str.split()


model_path = "/home/park/capstone/polyvore/saved_models/my_model.pt"
model = models.resnet18(pretrained=True)
num_classes = 5
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, num_classes)

model.load_state_dict(torch.load(model_path))
model.eval()



# image_path = "/home/park/capstone/polyvore/data/img/203/5.jpg"
# image = Image.open(image_path).convert('RGB')

# transform = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225]
#     )
# ])

# image = transform(image).unsqueeze(0)


# with torch.no_grad():
#     outputs = model(image)
#     _, predicted = torch.max(outputs.data, 1)

# # print("Predicted class index:", predicted.item())

acc = []
bottom = []
outer = []
shoes = []
top = []



"""
0 : acc
1 : bottom
2 : outer
3 : shoes
4 : top
""" 

file_path = '/home/park/capstone/polyvore/data/label/fashion_compatibility_prediction_musinsa.txt'

with open(file_path, 'r') as file:
    lines = file.readlines()

# 줄이 맨 앞이 1로 시작하지 않는 경우 해당 줄 제거
lines = [line for line in lines if line.startswith('1')]

# 파일을 쓰기 모드로 열고 새로운 내용을 저장
with open(file_path, 'w') as file:
    file.writelines(lines)


# file = open(file_path, 'w')


# new_content = ""

set_print = set()
for x in candidate :
    a, b = x.split('_')
    if b == 0 :
        continue
    image_path = "/home/park/capstone/polyvore/data/img/" + a + '/' + b + '.jpg'
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


    if predicted.item() == 1 :
        bottom.append(x)
    elif predicted.item() == 2 :
        outer.append(x)
    elif predicted.item() == 3 :
        shoes.append(x)
    elif predicted.item() == 4 :
        top.append(x)
    else :
        acc.append(x)

cnt = 0
print(bottom, outer, shoes, top, acc)

while cnt < 1500 :
    cnt += 1

    r1 = random.randint(0,len(bottom)-1)
    r2 = random.randint(0,len(outer)-1)
    r3 = random.randint(0,len(shoes)-1)
    r4 = random.randint(0,len(top)-1)
    r5 = random.randint(0,len(acc)-1) 

    if '0 '+ bottom[r1] + ' ' + outer[r2] + ' ' + shoes[r3] + ' ' + top[r4] + ' ' + acc[r5] in set_print :
        cnt -= 1
        continue
    set_print.add('0 '+ bottom[r1] + ' ' + outer[r2] + ' ' + shoes[r3] + ' ' + top[r4] + ' ' + acc[r5])


    new_content = '0 '+ bottom[r1] + ' ' + outer[r2] + ' ' + shoes[r3] + ' ' + top[r4] + ' ' + acc[r5] 
    with open(file_path, 'a') as file:
        file.write(new_content+ '\n')
    # print('0 '+ arr1[r1] + ' ' + arr2[r2] + ' ' + arr3[r3] + ' ' + arr4[r4] + ' ' + arr5[r5] )

cnt = 0

while cnt < 500 :
    cnt += 1

    r1 = random.randint(0,len(bottom)-1)
    r2 = random.randint(0,len(outer)-1)
    r3 = random.randint(0,len(shoes)-1)
    r4 = random.randint(0,len(top)-1)
    r5 = random.randint(0,len(acc)-1) 
    r6 = random.randint(0,len(acc)-1) 
    while r5 == r6 :
        r6 = random.randint(0,len(acc)-1) 

    if '0 '+ bottom[r1] + ' ' + outer[r2] + ' ' + shoes[r3] + ' ' + top[r4] + ' ' + acc[r5] + ' ' + acc[r6] in set_print :
        cnt -= 1
        continue
    set_print.add('0 '+ bottom[r1] + ' ' + outer[r2] + ' ' + shoes[r3] + ' ' + top[r4] + ' ' + acc[r5] + ' ' + acc[r6])


    new_content = '0 '+ bottom[r1] + ' ' + outer[r2] + ' ' + shoes[r3] + ' ' + top[r4] + ' ' + acc[r5] + ' ' + acc[r6] 
    with open(file_path, 'a') as file:
        file.write(new_content+ '\n')
    # print('0 '+ arr1[r1] + ' ' + arr2[r2] + ' ' + arr3[r3] + ' ' + arr4[r4] + ' ' + arr5[r5] )


cnt = 0
while cnt < 300 :
    cnt += 1

    r1 = random.randint(0,len(bottom)-1)
    # r2 = random.randint(0,len(outer)-1)
    r3 = random.randint(0,len(shoes)-1)
    r4 = random.randint(0,len(top)-1)
    r5 = random.randint(0,len(acc)-1) 

    if '0 '+ bottom[r1] + ' ' + shoes[r3] + ' ' + top[r4] + ' ' + acc[r5] in set_print :
        cnt -= 1
        continue
    set_print.add('0 '+ bottom[r1] + ' ' + shoes[r3] + ' ' + top[r4] + ' ' + acc[r5])


    new_content = '0 '+ bottom[r1]  + ' ' + shoes[r3] + ' ' + top[r4] + ' ' + acc[r5] 
    with open(file_path, 'a') as file:
        file.write(new_content+ '\n')
