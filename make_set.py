from rembg import remove
from PIL import Image
import os
import json
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import torchvision.models as models


model_path = "/home/park/capstone/polyvore/saved_models/my_model.pt"
model = models.resnet18(pretrained=True)
num_classes = 5
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, num_classes)

model.load_state_dict(torch.load(model_path))
model.eval()



json_path = '/home/park/capstone/polyvore/query.json'



# 이미지들이 있는 폴더 경로
folder_path = "/home/park/capstone/polyvore/results/"

with open(json_path, "r") as json_file:
    data = json.load(json_file)

# JSON 데이터 확인
for item in data:
    image_query = item["image_query"]
    text_query = item["text_query"]
    print("Image Query:", image_query)
    print("Text Query:", text_query)



final_set = []
clothes = set()
kind = [0,0,0,0,0]





if text_query == 'summer' or text_query == 'SUMMER' :
    clothes.add(2)
    print('summer time ')


for filename in os.listdir(folder_path):
    with open(folder_path + filename, 'rb') as i:
        print(filename[2:-4])
        if filename[2:-4]  in image_query :
            final_set.append(filename)

            image = Image.open(folder_path+filename).convert('RGB')

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

            clothes.add(predicted.item())
            kind[predicted.item()] = filename[2:]

print(final_set)

for filename in os.listdir(folder_path):
    with open(folder_path + filename, 'rb') as i:

        image = Image.open(folder_path+filename).convert('RGB')

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

        if predicted.item() in clothes:  
            print('aa')
            continue 
        elif predicted.item() not in clothes:  
            if  (text_query == ' summer' and predicted.item() == 2) : continue
            clothes.add(predicted.item())
            final_set.append(filename)
            kind[predicted.item()] = filename[2:]

### 누끼 따끼
print(final_set)

cnt = 1
for filename in os.listdir(folder_path):
    if filename not in final_set :
        print('bb')
        continue
    with open(folder_path + filename, 'rb') as i:
        with open('/home/park/capstone/polyvore/final_set/' + filename[2:], 'wb') as o:
            cnt += 1
            input = i.read()
            output = remove(input)
            o.write(output)


# 결과 이미지의 가로와 세로 크기
result_width = 400
result_height = 500

# 결과 이미지 생성
result_image = Image.new("RGB", (result_width, result_height), color=(255, 255, 204))



folder_path = "/home/park/capstone/polyvore/final_set/"


image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg') or f.endswith('.png')]

outer_overlay = ""
acc_overlay = ""

# 이미지들을 순회하며 배경 이미지에 합치기
for image_file in image_files:
    image_path = os.path.join(folder_path, image_file)
    # print(image_path)

    # 다른 이미지 열기
    overlay = Image.open(image_path)


    # overlay 이미지 크기 조정
    # overlay = overlay.resize((200, 200))  # 원하는 크기로 조정


    # 배경 이미지에 다른 이미지 합치기


    kind_clothes = image_path.split('/')[-1]

    if kind_clothes == kind[0] :
        acc_overlay = overlay

    elif kind_clothes == kind[1] :
        overlay = overlay.resize((250, 250))  # 원하는 크기로 조정
        result_image.paste(overlay, (150, 80), overlay)


    elif kind_clothes == kind[3] :
        overlay = overlay.resize((120, 120)) 
        result_image.paste(overlay, (200,330), overlay)

    elif kind_clothes == kind[4] :
        overlay = overlay.resize((250, 250))  # 원하는 크기로 조정
        result_image.paste(overlay, (10, 80), overlay)

    elif kind_clothes == kind[2] :
        outer_overlay = overlay
        # overlay = overlay.resize((170, 170))  # 원하는 크기로 조정
        # result_image.paste(overlay, (150, 300), overlay)

if kind[2] != 0 :
    outer_overlay = outer_overlay.resize((190, 190))  # 원하는 크기로 조정
    result_image.paste(outer_overlay, (120, 180), outer_overlay)


if kind[0] != 0 :
    acc_overlay = acc_overlay.resize((120, 120))  # 원하는 크기로 조정
    result_image.paste(acc_overlay, (10, 250), acc_overlay)

# 결과 이미지 저장
result_image.save('merged_image.jpg')


"""
0 : acc
1 : bottom
2 : outer
3 : shoes
4 : top
""" 




# # 폴더 내의 이미지들을 읽어서 합치기
# for filename in os.listdir(folder_path):
#     if filename.endswith(".jpg") or filename.endswith(".png"):
#         image_path = os.path.join(folder_path, filename)
#         image = Image.open(image_path)
#         image = image.resize((result_width, result_height), Image.ANTIALIAS)
#         result_image.paste(image, (0, 0))

# # 결과 이미지 저장
# result_image.save("결과 이미지.jpg")







# # 배경 이미지 파일 경로
# background_image_path = 'background.jpg'

# # 이미지들이 있는 폴더 경로
# folder_path = "/home/park/capstone/polyvore/final_set/"

# image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg') or f.endswith('.png')]



# # 배경 이미지 열기
# background_image = Image.open(background_image_path)


# # 이미지들을 순회하며 배경 이미지에 합치기
# for image_file in image_files:
#     image_path = os.path.join(folder_path, image_file)

#     # 다른 이미지 열기
#     overlay = Image.open(image_path)

#     # 배경 이미지에 다른 이미지 합치기
#     background_image.paste(overlay, (0, 0), overlay)

# # 결과 이미지 저장
# background_image.save('merged_image.jpg')