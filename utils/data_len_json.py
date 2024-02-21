# FILE_NAME 수정
FILE_NAME = 'train_filtered.json'

import json

with open(FILE_NAME, 'r') as f:
    data = json.load(f)

print(len(data))