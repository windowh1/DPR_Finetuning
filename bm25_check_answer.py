import sys
sys.path.append('./DPR')

from functools import partial
import json
import logging
from multiprocessing import Pool as ProcessPool
import pickle
from typing import List, Dict

import hydra
from omegaconf import DictConfig, OmegaConf
from dpr.options import setup_logger
from dpr.utils.tokenizers import SimpleTokenizer
from dpr.data.qa_validation import has_answer

logger = logging.getLogger()
setup_logger(logger)


def check_answer_from_meta(
    answers_and_docs,
    tokenizer,
    match_type,
) -> List[bool]:

    answers, docs = answers_and_docs

    hits = []

    for doc in docs:
        answer_found = False
        if has_answer(answers, doc["contents"], tokenizer, match_type):
            answer_found = True
        hits.append(answer_found)
    return hits


def calculate_matches_from_meta(
    answers_list: List[List[str]], 
    top_docs_list: List[List[Dict]],
    workers_num: int,
    match_type: str,
) -> List[List[bool]]:

    tok_opts = {}
    tokenizer = SimpleTokenizer(**tok_opts)

    processes = ProcessPool(processes=workers_num)
    logger.info("Matching answers in top docs...")
    get_hits_partial = partial(
        check_answer_from_meta,
        tokenizer=tokenizer,
        match_type=match_type,
    )

    answers_and_docs_list = zip(answers_list, top_docs_list)
    hits_list = processes.map(get_hits_partial, answers_and_docs_list)
    logger.info("Per question validation results len=%d", len(hits_list))

    return hits_list


def save_results_from_meta(
    questions: List[str],
    answers_list: List[List[str]],
    top_docs_list: List[List[Dict]],
    hits_list: List[List[bool]],
    out_file: str,
):
    # join docs text with the result ids, their questions and assigning has|no answer labels
    merged_data = []
    # assert len(per_question_hits) == len(questions) == len(answers)
    for i, q in enumerate(questions):
        answers = answers_list[i]
        top_docs = top_docs_list[i]
        hits = hits_list[i]

        results_item = {
            "question": q,
            "answers": answers,
            "ctxs": [
                {
                    "id": doc["id"],
                    "title": doc["titles"],
                    "text": doc["contents"],
                    "score": doc["scores"],
                    "has_answer": hits[j],
                }
                for j, doc in enumerate(top_docs)
            ],
        }

        merged_data.append(results_item)

    with open(out_file, "w") as writer:
        writer.write(json.dumps(merged_data, indent=4) + "\n")
    logger.info("Saved results * scores  to %s", out_file)



@hydra.main(config_path="myConf", config_name="bm25")
def main(cfg: DictConfig):

    logger.info("%s", OmegaConf.to_yaml(cfg))

    # load questions & answers
    questions = []
    answers_list = []

    if not cfg.qa_dataset:
        logger.warning("Please specify qa_dataset to use")
        return

    ds_key = cfg.qa_dataset
    logger.info("qa_dataset: %s", ds_key)
    
    qa_src = hydra.utils.instantiate(cfg.datasets[ds_key])
    qa_src.load_data()
    total_queries = len(qa_src)
    
    for i in range(total_queries):
        qa_sample = qa_src[i]
        question, answers = qa_sample.query, qa_sample.answers
        questions.append(question)
        answers_list.append(answers)
    logger.info("questions len %d", len(questions))

    # get top k results
    with open(cfg.out_path + 'top_docs_list_top.pickle', 'rb') as f:
        top_docs_list = pickle.load(f)

    hits_list = calculate_matches_from_meta(answers_list, top_docs_list, cfg.validation_workers, cfg.match)

    save_results_from_meta(questions, answers_list, top_docs_list, hits_list, cfg.out_path + 'retrieval_result_bm25')
        
if __name__ == "__main__":
    main()