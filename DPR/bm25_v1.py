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

    qa_set = pd.read_csv('/home/hyewonjeon/project/downloads/data/retriever/qas/nq-train.csv', sep='\t', header=None)
    all_queries = qa_set[0].to_list()

    ctx_src = hydra.utils.instantiate(cfg.ctx_sources[cfg.ctx_src])
    all_passages_dict = {}
    ctx_src.load_data_to(all_passages_dict)
    all_ctxs = [v.text for k, v in all_passages_dict.items()]
    
    tensorizer = get_bert_tensorizer_bm25(cfg)    
    tokenized_queries = [tensorizer.text_to_tensor(query) for query in all_queries]
    tokenized_ctxs = [tensorizer.text_to_tensor(ctx) for ctx in all_ctxs]

    bm25 = BM25Okapi(tokenized_ctxs)
    parallel = Parallel(n_jobs=-1, return_as="generator")
    top_k_ctx_id = parallel(delayed(process_query)(query, bm25) for query in tokenized_queries)

    with open('bm25.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(top_k_ctx_id)

if __name__ == "__main__":
    main()