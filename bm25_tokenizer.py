# pickle 사용?
import sys
sys.path.append('./DPR')

import logging
import math
import os
import pathlib
import pickle
import transformers
if transformers.__version__.startswith("4"):
    from transformers import BertTokenizer
else:
    from transformers.tokenization_bert import BertTokenizer

import hydra
from omegaconf import DictConfig

from dpr.utils.data_utils import Tensorizer
from dpr.models.hf_models import _add_special_tokens
from dpr.models.hf_models import get_bert_tokenizer
from dpr.options import setup_logger

logger = logging.getLogger()
setup_logger(logger)


class BertTensorizerBM25(Tensorizer):

    """
    dpr.models.hf_models.BertTensorizer과의 차이점
    - no padding, no sep token
    - tensor가 아닌 list 반환
    """

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


# class ListQueries(object):
#     def __init__(self, cfg):
#         self.cfg = cfg
    
#     def load(self):
#         if not self.cfg.qa_dataset:
#             logger.warning("Please specify qa_dataset to use")
#             return

#         qa_src = hydra.utils.instantiate(self.cfg.datasets[self.cfg.qa_dataset])
#         qa_src.load_data()
#         queries = [qa.query for qa in qa_src]

#         return queries


class ListContexts(object):
    def __init__(self, cfg):
        self.cfg = cfg
    
    def load(self):
        if not self.cfg.ctx_src:
            logger.warning("Please specify ctx_sources to use")
            return
        
        all_passages_dict = {}
        ctx_src = hydra.utils.instantiate(self.cfg.ctx_sources[self.cfg.ctx_src])
        ctx_src.load_data_to(all_passages_dict)
        contexts = [v.text for k, v in all_passages_dict.items()]

        return contexts


def ctxs_tokenizing(cfg, tensorizer, shard_ctxs):
    shard_tokens = []
    n = len(shard_ctxs)
    bsz = cfg.batch_size
    for batch_start in range(0, n, bsz):
        batch = shard_ctxs[batch_start: (batch_start + bsz if batch_start + bsz < n else n)]
        batch_tokens = [tensorizer.text_to_tensor(ctx) for ctx in batch]
        shard_tokens.append(batch_tokens)
    
    return shard_tokens


@hydra.main(config_path="myConf", config_name="bm25")
def main(cfg: DictConfig):

    tensorizer = get_bert_tensorizer_bm25(cfg)    

    logger.info("\nLoading contexts..")
    list_ctxs = ListContexts(cfg).load()
    
    total_size = len(list_ctxs)
    shard_size = math.ceil(total_size / cfg.num_shards)
    start_idx = cfg.shard_id * shard_size
    end_idx = start_idx + shard_size if start_idx + shard_size < total_size else total_size
    logger.info(
        "\nProducing tokens for passages range: %d to %d (out of total %d)",
        start_idx,
        end_idx,
        len(list_ctxs),
    )
    shard_ctxs = list_ctxs[start_idx:end_idx]
    shard_tokens = ctxs_tokenizing(cfg, tensorizer, shard_ctxs)

    file = cfg.out_file + "_" + str(cfg.shard_id)
    pathlib.Path(os.path.dirname(file)).mkdir(parents=True, exist_ok=True)
    logger.info("Writing results to %s" % file)
    with open(file, mode="wb") as f:
        pickle.dump(shard_tokens, f)


if __name__ == "__main__":
    main()