import sys
sys.path.append('./DPR')

import csv
import json
import logging
import pandas as pd
import os

import hydra
from omegaconf import DictConfig, OmegaConf
from dpr.options import setup_logger

logger = logging.getLogger()
setup_logger(logger)


def match_index(cfg):
    training_set_index_file = cfg.out_path + 'training_set_index.csv'

    if os.path.isfile(training_set_index_file):
        training_set_index = pd.read_csv(training_set_index_file, header=None)[0].to_list()
        return training_set_index

    else:
        training_set_index = []
        with open(cfg.training_set, 'r') as f:
            training_set = json.load(f)

        qa_set = hydra.utils.instantiate(cfg.datasets[cfg.qa_dataset])
        qa_set.load_data()

        for sample in training_set:
            training_set_index.append(qa_set[(qa_set[0] == sample["question"]) & (qa_set[1] == str(sample["answers"]))].index.tolist()[0])

        training_set_index_csv = [[0] for _ in range(len(training_set_index))]
        for i in range(len(training_set_index)):
            training_set_index_csv[i][0] = training_set_index[i]

        with open(training_set_index_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerows(training_set_index_csv)

        return training_set_index


def count_true_retrieval(cfg, retrieval_type):    
    cnt_list_file = cfg.out_path + 'true_retrieval_count_' + retrieval_type + '.csv'

    if os.path.isfile(cnt_list_file):
        cnt_list = pd.read_csv(cnt_list_file, header=None)[0].to_list()
        return cnt_list
    
    else:
        cnt_list = [] # 맞게 retrieve한 passage 개수
        
        if retrieval_type == 'dpr':
            with open(cfg.retrieval_result_dpr, 'r') as f:
                retrieval_result = json.load(f)
        else:
            with open(cfg.out_path + 'retrieval_result_bm25', 'r') as f:
                retrieval_result = json.load(f)
        
        for sample in retrieval_result:
            ctxs = sample["ctxs"]
            cnt = 0
            for i in range(len(ctxs)):
                ctx = ctxs[i]
                if ctx["has_answer"] == True:
                    cnt += 1
            cnt_list.append(cnt) 
        
        cnt_list_csv = [[0] for _ in range(len(cnt_list))]
        for i in range(len(cnt_list)):
            cnt_list_csv[i][0] = cnt_list[i]

        with open(cnt_list_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerows(cnt_list_csv)
        
        return cnt_list


# training_set filtering & hard_negative_ctxs 갈음
def filtering_data(cfg, training_set, retrieval_result, training_set_index, 
                   dpr_count, bm25_count, top_k=100, true_passages_num=0):
    filtered_data = []
    min_num_hard_neg = 100 # training_set(original)에서 retrieve한 passage 개수
    
    for i in range(top_k):
        dpr_cnt = dpr_count[i]
        bm25_cnt = bm25_count[i]
        
        if (dpr_cnt < bm25_cnt) and (dpr_cnt <= true_passages_num) and (i in training_set_index):
            retrieval_sample = retrieval_result[i]
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
            filtered_data.append(training_set[j])
            filtered_data[-1]["hard_negative_ctxs"] = hard_negative_ctxs
            
    with open(cfg.out_path + 'train_filtered' + '_top_' + str(top_k) + '_true_psg_' + str(true_passages_num) + '.json', 'w') as f:
        json.dump(filtered_data, f, indent=4)

    print("# of filtered data with true retrieval <=", true_passages_num, "among top", top_k, "passages:", len(filtered_data))

    return min_num_hard_neg # 가장 작은 hard negative ctxs 수



@hydra.main(config_path="myConf", config_name="bm25")
def main(cfg: DictConfig):

    logger.info("%s", OmegaConf.to_yaml(cfg))

    training_set_index = match_index(cfg)
    dpr_count = count_true_retrieval(cfg, 'dpr')
    bm25_count = count_true_retrieval(cfg, 'bm25')

    with open(cfg.training_set, 'r') as f:
        training_set = json.load(f)

    with open(cfg.retrieval_result_dpr, 'r') as f:
        retrieval_result = json.load(f)
    
    min_num_hard_neg = filtering_data(cfg, training_set, retrieval_result, training_set_index, 
                                      dpr_count, bm25_count, top_k=50, true_passages_num=1)
    
    print("minimum # of negative samples of filtered data:", min_num_hard_neg)



if __name__ == "__main__":
    main()