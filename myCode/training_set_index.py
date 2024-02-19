import csv
import json
import pandas as pd
import utils.path as path

with open(path.NQ_TRAINING_SET, 'r') as f:
    training_set = json.load(f)

qa_set = pd.read_csv(path.NQ_QA_SET, sep='\t', header=None)

# training_set의 qa_set에서의 index
def match_index(training_set, qa_set):
    training_set_index = []
    for sample in training_set:
        training_set_index.append(qa_set[(qa_set[0] == sample["question"]) & (qa_set[1] == str(sample["answers"]))].index.tolist()[0])
    return training_set_index

training_set_index = match_index(training_set, qa_set)

training_set_index_csv = [[0] for _ in range(len(training_set_index))]
for i in range(len(training_set_index)):
    training_set_index_csv[i][0] = training_set_index[i]

with open(path.MYCODE_DATA + 'training_set_index.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(training_set_index_csv)

