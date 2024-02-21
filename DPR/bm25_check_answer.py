import csv
from functools import partial
import json
import logging
from multiprocessing import Pool as ProcessPool
from typing import Tuple, List, Dict

import hydra
from omegaconf import DictConfig, OmegaConf
from dpr.options import setup_logger
from dpr.utils.tokenizers import SimpleTokenizer
from dpr.data.qa_validation import has_answer, has_answer_kilt, QAMatchStats

logger = logging.getLogger()
setup_logger(logger)

def check_answer(questions_answers_docs, tokenizer, match_type) -> List[bool]:
    """Search through all the top docs to see if they have any of the answers."""
    answers, doc_ids = questions_answers_docs # data type 변경

    global dpr_all_documents
    hits = []

    for i, doc_id in enumerate(doc_ids):
        doc = dpr_all_documents[doc_id]
        text = doc[0]

        answer_found = False
        if text is None:  # cannot find the document for some reason
            logger.warning("no doc in db")
            hits.append(False)
            continue
        if match_type == "kilt":
            if has_answer_kilt(answers, text):
                answer_found = True
        elif has_answer(answers, text, tokenizer, match_type):
            answer_found = True
        hits.append(answer_found)
    return hits


def calculate_matches(
    all_docs: Dict[object, Tuple[str, str]],
    answers: List[List[str]],
    closest_docs: List[List[object]], # data type 변경 
    workers_num: int,
    match_type: str,
) -> QAMatchStats:

    logger.info("all_docs size %d", len(all_docs))
    global dpr_all_documents
    dpr_all_documents = all_docs
    logger.info("dpr_all_documents size %d", len(dpr_all_documents))

    tok_opts = {}
    tokenizer = SimpleTokenizer(**tok_opts)

    processes = ProcessPool(processes=workers_num)
    logger.info("Matching answers in top docs...")
    get_score_partial = partial(check_answer, match_type=match_type, tokenizer=tokenizer)

    questions_answers_docs = zip(answers, closest_docs)
    scores = processes.map(get_score_partial, questions_answers_docs)

    logger.info("Per question validation results len=%d", len(scores))

    n_docs = len(closest_docs[0]) # data type 변경
    top_k_hits = [0] * n_docs
    for question_hits in scores:
        best_hit = next((i for i, x in enumerate(question_hits) if x), None)
        if best_hit is not None:
            top_k_hits[best_hit:] = [v + 1 for v in top_k_hits[best_hit:]]

    return QAMatchStats(top_k_hits, scores)


def validate(
    passages: Dict[object, Tuple[str, str]],
    answers: List[List[str]],
    result_ctx_ids: List[List[object]], # data type 변경
    workers_num: int,
    match_type: str,
) -> List[List[bool]]:
    logger.info("validating passages. size=%d", len(passages))
    match_stats = calculate_matches(passages, answers, result_ctx_ids, workers_num, match_type)
    top_k_hits = match_stats.top_k_hits

    logger.info("Validation results: top k documents hits %s", top_k_hits)
    top_k_hits = [v / len(result_ctx_ids) for v in top_k_hits]
    logger.info("Validation results: top k documents hits accuracy %s", top_k_hits)
    return match_stats.questions_doc_hits


def save_results(
    passages: Dict[object, Tuple[str, str]],
    questions: List[str],
    answers: List[List[str]],
    top_passages: List[List[object]], # data type 변경
    per_question_hits: List[List[bool]], # [각 question에 대한 [각 retrieved ctx의 hit 여부 (bool)]]
    out_file: str,
):
    # join passages text with the result ids, their questions and assigning has|no answer labels
    merged_data = []
    # assert len(per_question_hits) == len(questions) == len(answers)
    for i, q in enumerate(questions):
        q_answers = answers[i]
        results = top_passages[i]
        hits = per_question_hits[i]
        docs = [passages[doc_id] for doc_id in results]
        scores = ["0"] * len(docs) # score 기록 생략
        ctxs_num = len(hits)

        results_item = {
            "question": q,
            "answers": q_answers,
            "ctxs": [
                {
                    "id": results[c],
                    "title": docs[c][1],
                    "text": docs[c][0],
                    "score": scores[c],
                    "has_answer": hits[c],
                }
                for c in range(ctxs_num)
            ],
        }

        merged_data.append(results_item)

    with open(out_file, "w") as writer:
        writer.write(json.dumps(merged_data, indent=4) + "\n")
    logger.info("Saved results * scores  to %s", out_file)



def get_all_passages(ctx_sources):
    all_passages = {}
    for ctx_src in ctx_sources:
        ctx_src.load_data_to(all_passages)
        logger.info("Loaded ctx data: %d", len(all_passages))

    if len(all_passages) == 0:
        raise RuntimeError("No passages data found. Please specify ctx_file param properly.")
    return all_passages



@hydra.main(config_path="conf", config_name="dense_retriever")
def main(cfg: DictConfig):

    logger.info("%s", OmegaConf.to_yaml(cfg))

    # load questions & answers
    questions = []
    question_answers = []

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
        question_answers.append(answers)
    logger.info("questions len %d", len(questions))

    # get top k results
    top_results_and_scores = []
    with open('/Volumes/T7 Shield/project/DPR_Finetuning/bm25_test.csv', 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        for row in reader:
            top_results_and_scores.append(row)

    # load contexts 
    ctx_sources = []
    for ctx_src in cfg.ctx_datatsets:
        ctx_src = hydra.utils.instantiate(cfg.ctx_sources[ctx_src])
        ctx_sources.append(ctx_src)
        logger.info("ctx_sources: %s", type(ctx_src))

    all_passages = get_all_passages(ctx_sources)


    questions_doc_hits = validate(
        all_passages,
        question_answers,
        top_results_and_scores,
        cfg.validation_workers,
        cfg.match,
    )

    save_results(
        all_passages,
        questions,
        question_answers,
        top_results_and_scores,
        questions_doc_hits,
        cfg.out_file, 
    )
        
if __name__ == "__main__":
    main()

