import csv
from joblib import Parallel, delayed
import numpy as np
import pandas as pd

import transformers
if transformers.__version__.startswith("4"):
    from transformers import BertTokenizer
else:
    from transformers.tokenization_bert import BertTokenizer

import hydra
from omegaconf import DictConfig

from rank_bm25 import BM25Okapi

from dpr.utils.data_utils import Tensorizer
from dpr.models.hf_models import _add_special_tokens
from dpr.models.hf_models import get_bert_tokenizer

CPU_PER_TASKS = 8

# dpr.models.hf_models.BertTensorizer과의 차이점: no padding, no sep token, tensor가 아닌 list 반환
class BertTensorizerBM25(Tensorizer):
    def __init__(self, tokenizer: BertTokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def text_to_tensor(
        self,
        text: str,
        title: str = None,
        add_special_tokens: bool = True,
        apply_max_len: bool = True,
    ):
        text = text.strip()

        if title:
            token_ids = self.tokenizer.encode( # tokenize 및 token id로 encoding
                title,
                text_pair=text,
                add_special_tokens=add_special_tokens,
                max_length=self.max_length if apply_max_len else 10000,
                pad_to_max_length=False,
                truncation=True,
            )
        else:
            token_ids = self.tokenizer.encode(
                text,
                add_special_tokens=add_special_tokens,
                max_length=self.max_length if apply_max_len else 10000,
                pad_to_max_length=False,
                truncation=True,
            )

        seq_len = self.max_length
        if len(token_ids) > seq_len and apply_max_len:
            return token_ids[1:seq_len]
        else:
            return token_ids[1:-1] # trunc start and sep token


def get_bert_tensorizer_bm25(cfg):
    sequence_length = cfg.encoder.sequence_length
    pretrained_model_cfg = cfg.encoder.pretrained_model_cfg
    tokenizer = get_bert_tokenizer(pretrained_model_cfg, do_lower_case=cfg.do_lower_case)
    if cfg.special_tokens:
        _add_special_tokens(tokenizer, cfg.special_tokens)

    return BertTensorizerBM25(tokenizer, sequence_length)


class ListQueries(object):
    def __init__(self, path):
        self.path = path
    
    def load_data(self):
        qa_set = pd.read_csv(self.path, sep='\t', header=None)
        return qa_set[0].to_list()


class ListContexts(object):
    def __init__(self, cfg):
        self.cfg = cfg
    
    def load_data(self):
        ctx_src = hydra.utils.instantiate(self.cfg.ctx_sources[self.cfg.ctx_src])
        all_passages_dict = {}
        ctx_src.load_data_to(all_passages_dict)
        return [v.text for k, v in all_passages_dict.items()]

class TokenizedContexts(object):
    def __init__(self, tensorizer):
        self.tensorizer = tensorizer

    def load_data(self, list_ctxs, n_jobs):
        return Parallel(n_jobs=n_jobs, return_as="list")(delayed(self.tensorizer.text_to_tensor)(ctx) for ctx in list_ctxs)


def top_k_idx(doc_scores, k=100):
    partitioned_idx = np.argpartition(-doc_scores, k)[:k]
    sorted_idx = partitioned_idx[np.argsort(-doc_scores[partitioned_idx])]
    return sorted_idx


def process_query(query, bm25, k=100):
    doc_scores = bm25.get_scores(query)
    top_ctx_idx = top_k_idx(doc_scores, k)
    return top_ctx_idx + 1


@hydra.main(config_path="conf", config_name="gen_embs")
def main(cfg: DictConfig):

    tensorizer = get_bert_tensorizer_bm25(cfg)    
    print("\nloading queries...")
    list_queries = ListQueries('/home/hyewonjeon/project/downloads/data/retriever/qas/nq-train.csv').load_data()
    print("done")
    
    print("\ntokenizing queries...")
    tokenized_queries = [tensorizer.text_to_tensor(query) for query in list_queries]
    print("done")

    print("\nloading contexts...")
    list_ctxs = ListContexts(cfg).load_data()
    print("done")

    print("\ntokenizing contexts...")
    tokenized_ctxs = TokenizedContexts(tensorizer).load_data(list_ctxs, CPU_PER_TASKS)
    print("done")

    print("\nmaking BM25 object...")
    bm25 = BM25Okapi(tokenized_ctxs)
    print("done")
    
    print("\nsearching for top k ids...")
    top_k_ctx_id = Parallel(n_jobs=CPU_PER_TASKS, return_as="list")(delayed(process_query)(query, bm25) for query in tokenized_queries)
    print("done")

    print("\nwriting on the csv...")
    with open('bm25.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(top_k_ctx_id)
    print("done")
    print("\n=========================finish=========================")

if __name__ == "__main__":
    main()