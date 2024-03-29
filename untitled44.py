# -*- coding: utf-8 -*-
"""Untitled44.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/14GaJZ1mCT4EjvbnRuKpH1D3UUelXUcsJ
"""

# roboflow data 사용(Dataset에 맞게 링크 수정)
# 속도 증가를 위해 런타임 → 런타임 유형 변경 → GPU 설정
!curl -L -o robo.zip https://app.roboflow.com/ds/lHQjrwEQ5L?key=F7CY9HLA7d

!mkdir dataset

!mv robo.zip ./dataset

# Commented out IPython magic to ensure Python compatibility.
# %cd dataset
!unzip -oq robo.zip

# Commented out IPython magic to ensure Python compatibility.
# Yolov5 환경설정
# 초기화
# %cd /content
!git clone https://github.com/ultralytics/yolov5.git

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/yolov5/
!pip install -r requirements.txt

# Commented out IPython magic to ensure Python compatibility.
# %cat /content/dataset/data.yaml

# Commented out IPython magic to ensure Python compatibility.
# %cd /
from glob import glob
train_img_list = glob('/content/dataset/train/images/*.jpg')
val_img_list = glob('/content/dataset/valid/images/*.jpg')
print(len(train_img_list)), print(len(val_img_list))

with open('/content/dataset/train.txt', 'w') as f:
  f.write('\n'.join(train_img_list) + '\n')
with open('/content/dataset/val.txt', 'w') as f:
  f.write('\n'.join(val_img_list) + '\n')

#yaml 파일 업데이트
import yaml
with open('/content/dataset/data.yaml', 'r') as f:
  data = yaml.safe_load(f)
print(data)

data['train'] = '/content/dataset/train.txt'
data['val'] = '/content/dataset/val.txt'
with open('/content/dataset/data.yaml', 'w') as f:
  yaml.dump(data, f)
print(data)

# Commented out IPython magic to ensure Python compatibility.
# %cat /content/dataset/data.yaml

!cat /content/yolov5/models/yolov5s.yaml

# Commented out IPython magic to ensure Python compatibility.
# N회 반복학습 실시
# Batch, epochs는 적절히 수정
# name은 생성되는 폴더명
# /content/yolov5/runs/
# %cd /content/yolov5/
!python train.py --img 416 --batch 16 --epochs 500 --data /content/dataset/data.yaml --cfg ./models/yolov5s.yaml --weights yolov5s.pt --name 75QNED90_4