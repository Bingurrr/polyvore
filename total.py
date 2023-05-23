from flask import Flask, jsonify, request, send_file
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


@app.route('/recommand', methods=['POST'])
def recommand():
    try:
        print("@@@@@@@@")
        data = request.get_json()
        img = data.get('img')
        text = data.get('text')
        print(img)
        print(text)
        json_file_path = '/home/park/capstone/polyvore/query.json'

        # JSON 파일 열기
        with open(json_file_path, 'r') as file:
            data = json.load(file)

        # 데이터 수정
        data[0]['text_query'] = text
        data[0]['image_query'] = img
        # 수정된 데이터를 JSON 파일에 저장
        with open(json_file_path, 'w') as file:
            json.dump(data, file, indent=4)

        command = "./outfit_generation.sh && python3 make_set.py"
        
        # 명령어 실행
        subprocess.run(command, shell=True)

        out_path = "/home/park/capstone/polyvore/results"

        image_path = "/home/park/capstone/polyvore/merged_image.jpg"

        print("success")
        return send_file(image_path, mimetype='iamge/jpeg')
    except Exception as e:
        print(e)
        return 'Fail', 500


def append_to_musinsa_json(category, id):
    file_path = '/home/park/capstone/polyvore/data/label/musinsa.json'

    new_data = {
        "name": "clothes",
        "views": 100,
        "items": [
            {
                "index": 1,
                "name": "navy t-shirt",
                "price": 20,
                "likes": 20,
                "image": "",
                "categoryid": 1538
            }
        ],
        "image": "",
        "likes": 100,
        "date": "years",
        "set_url": "",
        "set_id": 0,
        "desc": ""
    }
    new_data['items']['categoryid'] = category
    new_data['set_id'] = str(id)

    with open(file_path, 'r+') as file:
        # Load existing JSON data
        json_data = json.load(file)

        # Append new data to the existing JSON data
        json_data.append(new_data)

        # Move the file pointer to the beginning
        file.seek(0)

        # Write the updated JSON data back to the file
        json.dump(json_data, file, indent=4)
        file.truncate()



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

        category_arr = [362,1035,54,1693,4054]
        append_to_musinsa_json(category_arr[predicted.item()], folder_num)


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




    
    # image_files = [os.path.join(out_path, f) for f in os.listdir(out_path) if f.endswith('.jpg')]
    # outputs = []

    # for path in image_files :
    #     with open(path, 'rb') as f :
    #         output_image = f.read()
    #         outputs.append(output_image)
    
    
    # #response = requests.post(url='http://server_ip:port/upload_images', files=outputs)

    # output_data = [data.decode('iso-8859-1') if isinstance(data, bytes) else data for data in outputs]
    # output_data = json.dumps({"output": output_data})
    # str = ""
    # for f in os.listdir(out_path) : 
    #     if f.endswith('.jpg') or f.endswith('.png') :
    #         str += f[:-4]
    

#     output_data = [f[:-4] for f in os.listdir(out_path) if f.endswith('.jpg') or f.endswith('.png')]
#     # print(type(output_data[0]))
    
#     # output_data = [data.decode('utf-16') if isinstance(data, bytes) else data for data in output_data]
#     # return jsonify(output_data)
#    # output_data = model(input_data)
#     return jsonify({'output': output_data})


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=9900)
