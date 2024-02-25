import sys
sys.path.append('./DPR')

import hydra
from omegaconf import DictConfig, OmegaConf
from dpr.options import setup_logger

import json
import logging
import pickle
# Traditional Lexical Models
from pyserini.search.lucene import LuceneSearcher

logger = logging.getLogger()
setup_logger(logger)



@hydra.main(config_path="myConf", config_name="bm25")
def main(cfg: DictConfig):

    logger.info("%s", OmegaConf.to_yaml(cfg))

    questions = []

    if not cfg.qa_dataset:
        logger.warning("Please specify qa_dataset to use")
        return

    ds_key = cfg.qa_dataset
    logger.info("qa_dataset: %s", ds_key)
    
    qa_src = hydra.utils.instantiate(cfg.datasets[ds_key])
    qa_src.load_data()
    
    for qa in qa_src:
        questions.append(qa.query)
    logger.info("questions len %d", len(questions))

    # Lucene index of Wikipedia with DPR 100-word splits
    searcher = LuceneSearcher(cfg.pyserini_index)
    # '/data2/hyewonjeon/project/DPR/lucene-index-wikipedia-dpr-100w-20210120-d1b9e6/'

    top_docs_list = []
    for question in questions[:5]:
        search_results = searcher.search(question)
        total_size = len(search_results)
        top_docs = []
        for i in range(cfg.retrieve_num):
            if cfg.mode == "bottom":
                i = total_size - i
            title_contents = search_results[i].lucene_document
            score = search_results[i].score
            raw = title_contents.get('raw')
            json_data = json.loads(raw)
            json_data['titles'], json_data['contents']= json_data['contents'].split(sep='\n')
            json_data['titles'] = json_data['titles'].strip("\"")
            json_data['scores'] = round(score, 5)
            top_docs.append(json_data)
        top_docs_list.append(top_docs)

    
    print(top_docs_list)

    with open(cfg.out_path + 'top_docs_list' + cfg.mode + '.pickle', 'wb') as f:
        pickle.dump(top_docs_list, f)


if __name__ == "__main__":
    main()