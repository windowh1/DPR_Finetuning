import hydra
from omegaconf import DictConfig

@hydra.main(config_path="conf", config_name="gen_embs")
def main(cfg: DictConfig):
    ctx_src = hydra.utils.instantiate(cfg.ctx_sources[cfg.ctx_src])
    all_passages_dict = {}
    ctx_src.load_data_to(all_passages_dict)
    all_passages = [(k, v) for k, v in all_passages_dict.items()]
    for i in range(5):
        print(all_passages[i])

if __name__ == "__main__":
    main()