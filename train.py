# train.py
import os
from networks import get_model
from solver import RedCNNSolver
from utils import get_loader, get_logger
from utils.transforms import *

import yaml

def train(args: dict):
    with open(f'config/{args.workframe}.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # args + yaml
    args_dict = vars(args)  # Namespace → dict
    overlap = set(args_dict).intersection(config)
    if overlap:
        print(f"[Warning] These keys from args will overwrite config: {overlap}")

    train_config = {**config, **args_dict}
    
    # Logger
    log_path = os.path.join(train_config['save_path'], "log")
    train_logger = get_logger(log_dir=log_path, name='train')
    val_logger = get_logger(log_dir=log_path, name='val')
    
    # Data loader
    train_loader = get_loader(mode='train',
                            data_path=train_config['data_path'],
                            transform=[Interpolation(scale=train_config['preprocess']['scale']),
                                       RandomCrop(train_config['train']['patch_size']),
                                       RandomFlip(),
                                       RandomRotate90(),
                                       Normalization(clip_min=train_config['preprocess']['norm_range_min'],
                                                     clip_max=train_config['preprocess']['norm_range_max']),
                                       ToTensor()])
    val_loader = get_loader(mode='val',
                          data_path=train_config['data_path'],
                          transform=[Interpolation(scale=train_config['preprocess']['scale']),
                                     RandomCrop(train_config['train']['patch_size']),
                                     Normalization(clip_min=train_config['preprocess']['norm_range_min'],
                                                   clip_max=train_config['preprocess']['norm_range_max']),
                                     ToTensor()])

    # Build model
    ModelClass = get_model(train_config['network']['type'])
    model = ModelClass()

    # Solver
    solver = RedCNNSolver(config=train_config, model=model,
                          train_loader=train_loader, train_logger=train_logger,
                          val_loader=val_loader, val_logger=val_logger)
    solver.train()