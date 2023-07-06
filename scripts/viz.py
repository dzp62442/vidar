# TRI-VIDAR - Copyright 2022 Toyota Research Institute.  All rights reserved.

import os
import sys
import fire
import torch

from display.displayer import Displayer
from vidar.core.trainer import Trainer
from vidar.core.wrapper import Wrapper
from vidar.utils.config import read_config, cfg_has, dataset_prefix
from vidar.utils.setup import setup_datasets, setup_dataloader
from vidar.utils.logging import pcolor


def display(**kwargs):

    os.environ['DIST_MODE'] = 'gpu' if torch.cuda.is_available() else 'cpu'

    cfg = 'configs/display/green_tractor_results.yaml'
    # cfg = 'configs/display/ddad_results.yaml'
    cfg = read_config(cfg, **kwargs)
    displayer = Displayer(cfg)
    displayer.show()


if __name__ == '__main__':
    fire.Fire(display)
