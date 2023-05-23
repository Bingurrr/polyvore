import random
import os
import json
import shutil
import random
import tensorflow as tf
import pickle as pkl
import numpy as np
import configuration
import polyvore_model_bi as polyvore_model


top = [1538, 4504, 921,54, 64,77, 83,607, 358,894]
bottom = [1035,1156,35, 4133,1850]
acc = [6350,553,439,1477, 337, 361,364,251, 362]
shoes = [1153,1693,2122, 216, 91]
outer = [263, 546, 2127, 719, 2922]

json_path = "/home/park/capstone/polyvore/data/label/musinsa.json"


FLAGS = tf.flags.FLAGS
FLAGS.rnn_type = 'lstm'
FLAGS.checkpoint_path = "/home/park/capstone/polyvore/model/final_model/model_final/model.ckpt-34865"


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

max_folder_num = find_num() - 200
# print(max_folder_num)

def get_set_id_and_index_top(json_file):
    with open(json_file, "r") as f:
        json_data = json.load(f)
    k = random.randint(0, max_folder_num)
    for item in json_data[k]['items']:
        if item['categoryid'] in top:
            return k+200, item['index'], json_data[k]

    return None, None, None

def get_set_id_and_index_bottom(json_file):
    with open(json_file, "r") as f:
        json_data = json.load(f)

    k = random.randint(0, max_folder_num)
    for item in json_data[random.randint(0, max_folder_num)]['items']:
        if item['categoryid'] in bottom:
            return k+200, item['index'], json_data[k]

    return None, None, None 

def get_set_id_and_index_acc(json_file):
    with open(json_file, "r") as f:
        json_data = json.load(f)
    k = random.randint(0, max_folder_num)
    for item in json_data[k]['items']:
        if item['categoryid'] in acc:
            return k+200, item['index'], json_data[k]

    return None, None, None

def get_set_id_and_index_shoes(json_file):
    with open(json_file, "r") as f:
        json_data = json.load(f)

    k = random.randint(0, max_folder_num)
    for item in json_data[k]['items']:
        if item['categoryid'] in shoes:
            return k+200, item['index'], json_data[k]

    return None, None, None

def get_set_id_and_index_outer(json_file):
    with open(json_file, "r") as f:
        json_data = json.load(f)

    k = random.randint(0, max_folder_num)
    for item in json_data[k]['items']:
        if item['categoryid'] in outer:
            return k+200, item['index'], json_data[k]

    return None, None, None


set_id, index, js = get_set_id_and_index_top(json_path)
# print(js['items'])

def predict(img_dir, json_file):
#   if os.path.isfile(FLAGS.feature_file):
#     print("Feature file already exist.")
#     return
  # Build the inference graph.
  FLAGS.image_dir = img_dir
  FLAGS.json_file = json_file
  g = tf.Graph()
  with g.as_default():
    model_config = configuration.ModelConfig()
    model_config.rnn_type = FLAGS.rnn_type
    model = polyvore_model.PolyvoreModel(model_config, mode="inference")
    model.build()
    saver = tf.train.Saver()
  g.finalize()
  sess = tf.Session(graph=g)
  saver.restore(sess, FLAGS.checkpoint_path)
  test_json = json.load(open(json_file))
  k = 0

  # Save image ids and features in a dictionary.
  test_features = dict()

  for image_set in test_json:
    set_id = image_set["set_id"]
    image_feat = []
    image_rnn_feat = []
    ids = []
    k = k + 1
    print(str(k) + " : " + set_id)
    for image in image_set["items"]:
      filename = os.path.join(FLAGS.image_dir, 
                              str(image["index"]) + ".jpg")
      with tf.gfile.GFile(filename, "r") as f:
        image_feed = f.read()

      [feat, rnn_feat] = sess.run([model.image_embeddings,
                                   model.rnn_image_embeddings],
                                  feed_dict={"image_feed:0": image_feed})
      
      image_name = set_id + "_" + str(image["index"])
      test_features[image_name] = dict()
      test_features[image_name]["image_feat"] = np.squeeze(feat)
      test_features[image_name]["image_rnn_feat"] = np.squeeze(rnn_feat)
  
#   with open(FLAGS.feature_file, "wb") as f:
#     pkl.dump(test_features, f)

  # Calculate compatibility score.
  compatibility_score = 0.0
  for image_set in test_json:
    set_id = image_set["set_id"]
    for image in image_set["items"]:
      image_name = set_id + "_" + str(image["index"])
      compatibility_score += np.dot(test_features[image_name]["image_feat"],
                                     test_features[image_name]["image_rnn_feat"])
  compatibility_score /= len(test_json)

  print("Compatibility score:", compatibility_score)
  return compatibility_score


def add_json_musinsa(idx) :
    add_clothes_path = '/home/park/capstone/polyvore/data/add_clothes/'
    img_path = '/home/park/capstone/polyvore/data/img/'

    shutil.rmtree(add_clothes_path)
    max_score = -999
    if idx == 0 :
        for i in range(5) :
            new_data = {
                "name": "clothes",
                "views": 100,
                "items": [
                ],
                "image": "",
                "likes": 100,
                "date": "years",
                "set_url": "",
                "set_id": 0,
                "desc": ""
            }
            input_data = {
                "price": 20,
                "image": "",
                "index": 0,
                "name": "",
                "categoryid": 362,
                "likes": 20
            }
            # input_data['categoryid'] = 362
            os.makedirs(os.path.join(add_clothes_path, str(i+1)))
            destination_folder = add_clothes_path + '/' + str(i+1) # os.path.join(add_clothes_path, str(i+1))
            set_id, index, js = get_set_id_and_index_top(json_path)
            # print(new_data["items"])
            cnt = 0
            # input_data['index'] = cnt
            # new_data["items"].append(input_data)
            if index != None :
                new_data["items"].append(js['items'][index-1])
                new_data["items"][cnt]['index'] = cnt+1
                source_folder = img_path + str(set_id) + '/' + str(cnt+1) + '.jpg'
                shutil.copy(source_folder, destination_folder)
                cnt += 1

            set_id, index, js = get_set_id_and_index_outer(json_path)
            if index != None :
                new_data["items"].append(js['items'][index-1])
                new_data["items"][cnt]['index'] = cnt+1
                source_folder = img_path + str(set_id) + '/' + str(cnt+1) + '.jpg'
                shutil.copy(source_folder, destination_folder)
                cnt += 1

            set_id, index, js = get_set_id_and_index_bottom(json_path)
            if index != None :
                new_data["items"].append(js['items'][index-1])
                new_data["items"][cnt]['index'] = cnt+1
                source_folder = img_path + str(set_id) + '/' + str(cnt+1) + '.jpg'
                shutil.copy(source_folder, destination_folder)
                cnt += 1

            set_id, index, js = get_set_id_and_index_shoes(json_path)
            if index != None :
                new_data["items"].append(js['items'][index-1])
                new_data["items"][cnt]['index'] = cnt+1
                source_folder = img_path + str(set_id) + '/' + str(cnt+1) + '.jpg'
                shutil.copy(source_folder, destination_folder)
                cnt += 1

            new_data['set_id'] = str(i+1)

            new_data = [new_data]
            print(new_data)
            with open('new_data' + str(i) + '.json', 'w') as f:
                json.dump(new_data, f, indent=4)

            if cnt >= 2 :
                s = predict('/home/park/capstone/polyvore/data/add_clothes/'+str(i+1), '/home/park/capstone/polyvore/polyvore/new_data' + str(i)+'.json')
            print(s)

            if s > max_score and cnt >= 2:
                s = max_score
                add_data = new_data[0]
                add_address = '/home/park/capstone/polyvore/data/add_clothes/' +str(i+1)

        existing_data = []
        with open("/home/park/capstone/polyvore/data/label/musinsa.json", "r") as f:
            existing_data = json.load(f)

        # 기존 데이터에 새로운 데이터 추가
        add_data['set_id'] = str(find_num()+1)
        existing_data.append(add_data)

        # 수정된 데이터 저장
        with open("/home/park/capstone/polyvore/data/label/musinsa.json", "w") as f:
            json.dump(existing_data, f, indent=4)

        
        dest = '/home/park/capstone/polyvore/data/img'
        shutil.move(add_address, os.path.join(dest, str(find_num()+1)))

    elif idx == 1 :
        for i in range(5) :
            new_data = {
                "name": "clothes",
                "views": 100,
                "items": [
                ],
                "image": "",
                "likes": 100,
                "date": "years",
                "set_url": "",
                "set_id": 0,
                "desc": ""
            }
            input_data = {
                "price": 20,
                "image": "",
                "index": 0,
                "name": "",
                "categoryid": 894,
                "likes": 20
            }
            input_data['categoryid'] = 1035
            os.makedirs(os.path.join(add_clothes_path, str(i+1)))
            destination_folder = os.path.join(add_clothes_path, str(i+1))
            set_id, index, js = get_set_id_and_index_top(json_path)
            # print(new_data["items"])
            cnt = 0
            input_data['index'] = cnt
            new_data["items"].append(input_data)
            if index != None :
                new_data["items"].append(js['items'][index-1])
                new_data["items"][cnt]['index'] = cnt+1
                source_folder = img_path + str(set_id) + '/' + str(cnt+1) + '.jpg'
                shutil.copy(source_folder, destination_folder)
                cnt += 1

            set_id, index, js = get_set_id_and_index_outer(json_path)
            if index != None :
                new_data["items"].append(js['items'][index-1])
                new_data["items"][cnt]['index'] = cnt+1
                source_folder = img_path + str(set_id) + '/' + str(cnt+1) + '.jpg'
                shutil.copy(source_folder, destination_folder)
                cnt += 1

            set_id, index, js = get_set_id_and_index_acc(json_path)
            if index != None :
                new_data["items"].append(js['items'][index-1])
                new_data["items"][cnt]['index'] = cnt+1
                source_folder = img_path + str(set_id) + '/' + str(cnt+1) + '.jpg'
                shutil.copy(source_folder, destination_folder)
                cnt += 1

            set_id, index, js = get_set_id_and_index_shoes(json_path)
            if index != None :
                new_data["items"].append(js['items'][index-1])
                new_data["items"][cnt]['index'] = cnt+1
                source_folder = img_path + str(set_id) + '/' + str(cnt+1) + '.jpg'
                shutil.copy(source_folder, destination_folder)
                cnt += 1
                
            new_data['set_id'] = str(i+1)

            new_data = [new_data]
            print(new_data)
            with open('new_data' + str(i) + '.json', 'w') as f:
                json.dump(new_data, f, indent=4)

            if cnt >= 3 :
                s = predict('/home/park/capstone/polyvore/data/add_clothes/'+str(i+1), '/home/park/capstone/polyvore/polyvore/new_data' + str(i)+'.json')
            print(s)

            if s > max_score and cnt >= 2:
                s = max_score
                add_data = new_data[0]
                add_address = '/home/park/capstone/polyvore/data/add_clothes/' +str(i+1)

        existing_data = []
        with open("/home/park/capstone/polyvore/data/label/musinsa.json", "r") as f:
            existing_data = json.load(f)

        # 기존 데이터에 새로운 데이터 추가
        add_data['set_id'] = str(find_num()+1)
        existing_data.append(add_data)

        # 수정된 데이터 저장
        with open("/home/park/capstone/polyvore/data/label/musinsa.json", "w") as f:
            json.dump(existing_data, f, indent=4)

        
        dest = '/home/park/capstone/polyvore/data/img'
        shutil.move(add_address, os.path.join(dest, str(find_num()+1)))

    elif idx == 2 :
        for i in range(5) :
            new_data = {
                "name": "clothes",
                "views": 100,
                "items": [
                ],
                "image": "",
                "likes": 100,
                "date": "years",
                "set_url": "",
                "set_id": 0,
                "desc": ""
            }
            input_data = {
                "price": 20,
                "image": "",
                "index": 0,
                "name": "",
                "categoryid": 894,
                "likes": 20
            }
            input_data['categoryid'] = 263
            os.makedirs(os.path.join(add_clothes_path, str(i+1)))
            destination_folder = os.path.join(add_clothes_path, str(i+1))
            set_id, index, js = get_set_id_and_index_top(json_path)
            # print(new_data["items"])
            cnt = 0
            input_data['index'] = cnt
            new_data["items"].append(input_data)
            if index != None :
                new_data["items"].append(js['items'][index-1])
                new_data["items"][cnt]['index'] = cnt+1
                source_folder = img_path + str(set_id) + '/' + str(cnt+1) + '.jpg'
                shutil.copy(source_folder, destination_folder)
                cnt += 1

            set_id, index, js = get_set_id_and_index_bottom(json_path)
            if index != None :
                new_data["items"].append(js['items'][index-1])
                new_data["items"][cnt]['index'] = cnt+1
                source_folder = img_path + str(set_id) + '/' + str(cnt+1) + '.jpg'
                shutil.copy(source_folder, destination_folder)
                cnt += 1

            set_id, index, js = get_set_id_and_index_acc(json_path)
            if index != None :
                new_data["items"].append(js['items'][index-1])
                new_data["items"][cnt]['index'] = cnt+1
                source_folder = img_path + str(set_id) + '/' + str(cnt+1) + '.jpg'
                shutil.copy(source_folder, destination_folder)
                cnt += 1

            set_id, index, js = get_set_id_and_index_shoes(json_path)
            if index != None :
                new_data["items"].append(js['items'][index-1])
                new_data["items"][cnt]['index'] = cnt+1
                source_folder = img_path + str(set_id) + '/' + str(cnt+1) + '.jpg'
                shutil.copy(source_folder, destination_folder)
                cnt += 1
                
            new_data['set_id'] = str(i+1)

            new_data = [new_data]
            print(new_data)
            with open('new_data' + str(i) + '.json', 'w') as f:
                json.dump(new_data, f, indent=4)

            if cnt >= 3 :
                s = predict('/home/park/capstone/polyvore/data/add_clothes/'+str(i+1), '/home/park/capstone/polyvore/polyvore/new_data' + str(i)+'.json')
            print(s)

            if s > max_score and cnt >= 2:
                s = max_score
                add_data = new_data[0]
                add_address = '/home/park/capstone/polyvore/data/add_clothes/' +str(i+1)

        existing_data = []
        with open("/home/park/capstone/polyvore/data/label/musinsa.json", "r") as f:
            existing_data = json.load(f)

        # 기존 데이터에 새로운 데이터 추가
        add_data['set_id'] = str(find_num()+1)
        existing_data.append(add_data)

        # 수정된 데이터 저장
        with open("/home/park/capstone/polyvore/data/label/musinsa.json", "w") as f:
            json.dump(existing_data, f, indent=4)

        
        dest = '/home/park/capstone/polyvore/data/img'
        shutil.move(add_address, os.path.join(dest, str(find_num()+1)))

    elif idx == 3 :
        for i in range(5) :
            new_data = {
                "name": "clothes",
                "views": 100,
                "items": [
                ],
                "image": "",
                "likes": 100,
                "date": "years",
                "set_url": "",
                "set_id": 0,
                "desc": ""
            }
            input_data = {
                "price": 20,
                "image": "",
                "index": 0,
                "name": "",
                "categoryid": 894,
                "likes": 20
            }
            input_data['categoryid'] = 1693
            os.makedirs(os.path.join(add_clothes_path, str(i+1)))
            destination_folder = os.path.join(add_clothes_path, str(i+1))
            set_id, index, js = get_set_id_and_index_top(json_path)
            # print(new_data["items"])
            cnt = 0
            input_data['index'] = cnt
            new_data["items"].append(input_data)
            if index != None :
                new_data["items"].append(js['items'][index-1])
                new_data["items"][cnt]['index'] = cnt+1
                source_folder = img_path + str(set_id) + '/' + str(cnt+1) + '.jpg'
                shutil.copy(source_folder, destination_folder)
                cnt += 1

            set_id, index, js = get_set_id_and_index_outer(json_path)
            if index != None :
                new_data["items"].append(js['items'][index-1])
                new_data["items"][cnt]['index'] = cnt+1
                source_folder = img_path + str(set_id) + '/' + str(cnt+1) + '.jpg'
                shutil.copy(source_folder, destination_folder)
                cnt += 1

            set_id, index, js = get_set_id_and_index_bottom(json_path)
            if index != None :
                new_data["items"].append(js['items'][index-1])
                new_data["items"][cnt]['index'] = cnt+1
                source_folder = img_path + str(set_id) + '/' + str(cnt+1) + '.jpg'
                shutil.copy(source_folder, destination_folder)
                cnt += 1

            set_id, index, js = get_set_id_and_index_acc(json_path)
            if index != None :
                new_data["items"].append(js['items'][index-1])
                new_data["items"][cnt]['index'] = cnt+1
                source_folder = img_path + str(set_id) + '/' + str(cnt+1) + '.jpg'
                shutil.copy(source_folder, destination_folder)
                cnt += 1
                
            new_data['set_id'] = str(i+1)

            new_data = [new_data]
            print(new_data)
            with open('new_data' + str(i) + '.json', 'w') as f:
                json.dump(new_data, f, indent=4)

            if cnt >= 2 :
                s = predict('/home/park/capstone/polyvore/data/add_clothes/'+str(i+1), '/home/park/capstone/polyvore/polyvore/new_data' + str(i)+'.json')
            print(s)

            if s > max_score and cnt >= 2:
                s = max_score
                add_data = new_data[0]
                add_address = '/home/park/capstone/polyvore/data/add_clothes/' +str(i+1)

        existing_data = []
        with open("/home/park/capstone/polyvore/data/label/musinsa.json", "r") as f:
            existing_data = json.load(f)

        # 기존 데이터에 새로운 데이터 추가
        add_data['set_id'] = str(find_num()+1)
        existing_data.append(add_data)

        # 수정된 데이터 저장
        with open("/home/park/capstone/polyvore/data/label/musinsa.json", "w") as f:
            json.dump(existing_data, f, indent=4)

        
        dest = '/home/park/capstone/polyvore/data/img'
        shutil.move(add_address, os.path.join(dest, str(find_num()+1)))

    else :
        for i in range(5) :
            new_data = {
                "name": "clothes",
                "views": 100,
                "items": [
                ],
                "image": "",
                "likes": 100,
                "date": "years",
                "set_url": "",
                "set_id": 0,
                "desc": ""
            }
            input_data = {
                "price": 20,
                "image": "",
                "index": 0,
                "name": "",
                "categoryid": 894,
                "likes": 20
            }
            input_data['categoryid'] = 4504
            os.makedirs(os.path.join(add_clothes_path, str(i+1)))
            destination_folder = os.path.join(add_clothes_path, str(i+1))
            set_id, index, js = get_set_id_and_index_bottom(json_path)
            # print(new_data["items"])
            cnt = 0
            input_data['index'] = cnt
            new_data["items"].append(input_data)
            if index != None :
                new_data["items"].append(js['items'][index-1])
                new_data["items"][cnt]['index'] = cnt+1
                source_folder = img_path + str(set_id) + '/' + str(cnt+1) + '.jpg'
                shutil.copy(source_folder, destination_folder)
                cnt += 1

            set_id, index, js = get_set_id_and_index_outer(json_path)
            if index != None :
                new_data["items"].append(js['items'][index-1])
                new_data["items"][cnt]['index'] = cnt+1
                source_folder = img_path + str(set_id) + '/' + str(cnt+1) + '.jpg'
                shutil.copy(source_folder, destination_folder)
                cnt += 1

            set_id, index, js = get_set_id_and_index_acc(json_path)
            if index != None :
                new_data["items"].append(js['items'][index-1])
                new_data["items"][cnt]['index'] = cnt+1
                source_folder = img_path + str(set_id) + '/' + str(cnt+1) + '.jpg'
                shutil.copy(source_folder, destination_folder)
                cnt += 1

            set_id, index, js = get_set_id_and_index_shoes(json_path)
            if index != None :
                new_data["items"].append(js['items'][index-1])
                new_data["items"][cnt]['index'] = cnt+1
                source_folder = img_path + str(set_id) + '/' + str(cnt+1) + '.jpg'
                shutil.copy(source_folder, destination_folder)
                cnt += 1
                
            new_data['set_id'] = str(i+1)

            new_data = [new_data]
            print(new_data)
            with open('new_data' + str(i) + '.json', 'w') as f:
                json.dump(new_data, f, indent=4)

            if cnt >= 3 :
                s = predict('/home/park/capstone/polyvore/data/add_clothes/'+str(i+1), '/home/park/capstone/polyvore/polyvore/new_data' + str(i)+'.json')
            print(s)

            if s > max_score and cnt >= 2:
                s = max_score
                add_data = new_data[0]
                add_address = '/home/park/capstone/polyvore/data/add_clothes/' +str(i+1)

        existing_data = []
        with open("/home/park/capstone/polyvore/data/label/musinsa.json", "r") as f:
            existing_data = json.load(f)

        # 기존 데이터에 새로운 데이터 추가
        add_data['set_id'] = str(find_num()+1)
        existing_data.append(add_data)

        # 수정된 데이터 저장
        with open("/home/park/capstone/polyvore/data/label/musinsa.json", "w") as f:
            json.dump(existing_data, f, indent=4)

        
        dest = '/home/park/capstone/polyvore/data/img'
        shutil.move(add_address, os.path.join(dest, str(find_num()+1)))


add_json_musinsa(0) 

            

# if predicted.item() == 0 :
#     response = "acc"
# elif predicted.item() == 1 :
#     response = "bottom"
# elif predicted.item() == 2 :
#     response = "outer"
# elif predicted.item() == 3 :
#     response = "shoes"
# else :
#     response = "top"



# print(set_id, index)

# import shutil

# def copy_file(source_path, destination_path):
#     shutil.copyfile(source_path, os.path.join(destination_path, os.path.basename(source_path)))


# source_folder = "/home/park/capstone/polyvore/data/label"
# destination_folder = "/home/park/capstone/polyvore/data/copy"

# for file in os.listdir(source_folder):
#     if file.startswith("set_id") and file.endswith("index"):
#         copy_file(os.path.join(source_folder, file), destination_folder)
