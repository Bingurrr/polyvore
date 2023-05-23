from flask import Flask, jsonify, request, send_file
import json, subprocess, requests, os



json_file_path = '/home/park/capstone/polyvore/query.json'

# JSON 파일 열기
with open(json_file_path, 'r') as file:
    data = json.load(file)

# 데이터 수정
print(data[0]['text_query'])
print(type(data[0]['image_query']))