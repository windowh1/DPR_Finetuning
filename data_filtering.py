import sys
sys.path.append('./DPR')

import csv
import json
import pandas as pd
import os

import utils.path as path
import hyperparameter as hp

with open(path.NQ_TRAINING_SET, 'r') as f:
    training_set = json.load(f)

with open(path.NQ_RETRIEVAL_RESULT, 'r') as f:
    retrieval_result = json.load(f)

qa_set = pd.read_csv(path.NQ_QA_SET, sep='\t', header=None)

training_set_index = pd.read_csv(path.TRAINING_SET_INDEX, header=None)[0].to_list()

# retrieve 결과 평가
# TODO: bm25 결과 반영
def count_true_retrieval(retrieval_result, training_set_index, top_k=100, true_passages_num=0):
    
    cnt_list = [] # 맞게 retrieve한 passage 개수
    for i in range(len(retrieval_result)):
        sample = retrieval_result[i]
        cnt = 0
        for k in range(top_k):
            ctx = sample["ctxs"][k]
            if ctx["has_answer"] == True:
                cnt += 1
        if (cnt <= true_passages_num) and (i in training_set_index):
            cnt = 0
        elif (cnt <= true_passages_num) and (i not in training_set_index):
            cnt = -1
        elif i not in training_set_index:
            cnt = -2
        cnt_list.append(cnt) 
    
    print("# of training samples with true retrieval <=", true_passages_num, "among top", top_k, "passages:", cnt_list.count(0))
        # 맞게 retrieve한 passage 개수가 true_passages_num 이하이고, training set에 포함되어 있는 경우
    
    # Additional infos
    # print("# of non-training samples with true retrieval <=", true_passages_num, "among top", top_k, "passages:", cnt_list.count(-1))
    #     # 맞게 retrieve한 passage 개수가 true_passages_num 이하이고, training set에 포함되어 있지 않은 경우
    # print("# of non-training samples with true retrieval >", true_passages_num, "among top", top_k, "passages:", cnt_list.count(-2))
    #     # 맞게 retrieve한 passage 개수가 true_passages_num 초과이나, training set에 포함되어 있지 않은 경우
    cnt_list_csv = [[0] for _ in range(len(cnt_list))]
    for i in range(len(cnt_list)):
        cnt_list_csv[i][0] = cnt_list[i]

    with open(path.MY_DATA + 'true_retrieval_count_top_' + str(top_k) + '_true_psg_' + str(true_passages_num) + '.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(cnt_list_csv)
    
    return cnt_list


# training_set filtering & hard_negative_ctxs 갈음
def filtering_training_set(training_set, retrieval_result, training_set_index, top_k=100, true_passages_num=0):
    
    cnt_list = pd.read_csv(path.MY_DATA + 'true_retrieval_count_top_' + str(top_k) + '_true_psg_' + str(true_passages_num) + '.csv', header=None)[0].to_list()
    min_num_hard_neg = 100 # training_set(original)에서 retrieve한 passage 개수

    new_training_set = []
    for i in range(len(cnt_list)):
        retrieval_sample = retrieval_result[i]
        if cnt_list[i] == 0: # 맞게 retrieve한 passage 개수가 true_passages_num 이하이고, training set에 포함되어 있는 경우
            hard_negative_ctxs = []
            for ctx in retrieval_sample["ctxs"]:
                if ctx["has_answer"] == False:
                    hard_negative_ctxs.append(
                        {"title"        : ctx["title"], 
                        "text"         : ctx["text"], 
                        "score"        : 0, 
                        "title_score"  : 0, 
                        "passage_id"   : ctx["id"]})
            if len(hard_negative_ctxs) < min_num_hard_neg:
                min_num_hard_neg = len(hard_negative_ctxs)
            j = training_set_index.index(i)  
            new_training_set.append(training_set[j])
            new_training_set[-1]["hard_negative_ctxs"] = hard_negative_ctxs
            
    with open(path.MY_DATA + 'train_filtered' + '_top_' + str(top_k) + '_true_psg_' + str(true_passages_num) + '.json', 'w') as f:
        json.dump(new_training_set, f, indent=4)
    
    return min_num_hard_neg # 가장 작은 hard negative ctxs 수