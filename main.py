import logging
import os
import random

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf


logger = logging.getLogger(__name__)
OmegaConf.register_new_resolver("calc", eval, replace=True)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _add_file_logging(log_path: str) -> None:
    """Attach a FileHandler to the root logger so all output is persisted.

    Hydra changes cwd to the run directory before calling main(), so
    ``log_path`` can be a relative path like ``train.log``.
    The handler uses the same format as Hydra's console handler.
    """
    root = logging.getLogger()
    # Avoid duplicate file handlers if this function is called more than once
    for h in root.handlers:
        if isinstance(h, logging.FileHandler) and h.baseFilename == os.path.abspath(log_path):
            return
    fmt = logging.Formatter(
        "%(asctime)s (%(levelname)s): %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(logging.DEBUG)
    root.addHandler(fh)
    root.info(f"File logging enabled → {os.path.abspath(log_path)}")


@hydra.main(config_path="configs", config_name="base", version_base="1.1")
def main(cfg: DictConfig) -> None:
    # Hydra has already set cwd to results_dir/expname at this point.
    # Save the full resolved config for reproducibility.
    OmegaConf.save(cfg, "hparams.yaml")

    # Attach Python-level file logging (complements the tee in the shell script)
    _add_file_logging("train.log")
    set_global_seed(int(cfg.seed))
    logger.info(f"Global seed set to {int(cfg.seed)}")

    hydra.core.global_hydra.GlobalHydra.instance().clear()
    reinl = hydra.utils.instantiate(
        cfg.pipeline,
        model_suite=cfg.model,
        reward=cfg.reward,
        logger=cfg.logger,
    )
    reinl.run_rl()


if __name__ == '__main__':
    main()
