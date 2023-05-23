#!/bin/bash
CHECKPOINT_DIR="model/final_model/model.ckpt-34865"

# activate_script="/home/park/anaconda3/bin/activate"
# source "$activate_script"

source activate env

# Run inference on images.
python polyvore/set_generation.py \
  --checkpoint_path=${CHECKPOINT_DIR} \
  --image_dir="data/img/" \
  --feature_file="data/features/test_features_musinsa.pkl" \
  --query_file="query.json" \
  --word_dict_file="data/final_word_dict.txt" \
  --result_dir="results/"



# activate_script="/path/to/your/virtualenv/bin/activate"

# # 가상 환경 활성화
# source "$activate_script"

# # 가상 환경이 활성화된 상태에서 실행될 명령들
# python your_script.py

# 가상 환경 비활성화