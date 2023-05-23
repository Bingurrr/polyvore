#!/bin/bash
CHECKPOINT_DIR="model/final_model/model.ckpt-34865"

source activate env

python polyvore/fashion_compatibility.py \
  --checkpoint_path=${CHECKPOINT_DIR} \
  --label_file="data/label/fashion_compatibility_prediction_musinsa.txt" \
  --feature_file="data/features/test_features_musinsa.pkl" \
  --rnn_type="lstm" \
  --direction="2" \
  --result_file="fashion_compatibility_musinsa.pkl"
