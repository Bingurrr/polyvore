from flask import Flask, jsonify, request
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import torchvision.models as models
import json, subprocess, requests, os
import shutil

app = Flask(__name__)


"""
0 : acc
1 : bottom
2 : outer
3 : shoes
4 : top
""" 

def find_num() :

    folder_path = '/home/park/capstone/polyvore/data/img'

    # 폴더 내 파일 목록 가져오기
    files = os.listdir(folder_path)

    max_number = float('-inf')  # 초기 최대 값 설정

    # 파일 목록을 순회하며 가장 큰 숫자 찾기
    for file in files:
        # 파일 이름에서 숫자 추출
        file_name = os.path.splitext(file)[0]  # 확장자 제거
        try:
            number = int(file_name)
            max_number = max(max_number, number)
        except ValueError:
            continue  # 숫자로 변환할 수 없는 경우 건너뛰기

    # 최대 숫자를 가진 파일 이름 반환
    max_file_name = str(max_number) if max_number != float('-inf') else None
    return int(max_file_name)


def file_save(request):
    try:
        # Check if the 'file' key is in the request.files dictionary
        if 'file' not in request.files:
            return 'No file uploaded', 400

        file = request.files['file']        
        # Save the file to a specific location
        # print(type(file))
        file.save('/home/park/capstone/polyvore/data/tmp/file.jpg')


        model_path = "/home/park/capstone/polyvore/saved_models/my_model.pt"
        model = models.resnet18(pretrained=True)
        num_classes = 5
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, num_classes)

        model.load_state_dict(torch.load(model_path))
        model.eval()

        
        image_path = "/home/park/capstone/polyvore/data/tmp/file.jpg"
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

        
        folder_num = find_num() + 1

        file = Image.open("/home/park/capstone/polyvore/data/tmp/file.jpg")

        p = "/home/park/capstone/polyvore/data/img/" + str(folder_num) 
        os.makedirs(p)

        shutil.move(image_path, p + '/' +str(predicted.item()) + '.jpg')
        # file = request.files['file']
        # file_path = p+ '/' + str(predicted.item())  +'.jpg'      
        # file.save(file_path)

        # Process the file and return the response

        if predicted.item() == 0 :
            response = "acc"
        elif predicted.item() == 1 :
            response = "bottom"
        elif predicted.item() == 2 :
            response = "outer"
        elif predicted.item() == 3 :
            response = "shoes"
        else :
            response = "top"

        return response
    except Exception as e:
        print(e)
        return 'Fail', 500




@app.route('/classify', methods=['POST'])
def predict():
    output_data = file_save(request)    
    return output_data

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=9900)


