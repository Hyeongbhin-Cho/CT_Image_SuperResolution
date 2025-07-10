# eval.py
import os
from networks import get_model
from solver import RedCNNSolver
from utils import get_loader, get_logger
from utils.transforms import *

import yaml

def evaluate(args: dict):
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
    eval_logger = get_logger(log_dir=log_path, name='eval')
    
    # Data loader
    eval_loader = get_loader(mode='eval',
                            data_path=train_config['data_path'],
                            transform=[Interpolation(scale=train_config['preprocess']['scale']),
                                        Normalization(clip_min=train_config['preprocess']['norm_range_min'],
                                                    clip_max=train_config['preprocess']['norm_range_max']),
                                        ToTensor()])

    # Build model
    ModelClass = get_model(train_config['network']['type'])
    model = ModelClass()

    # Solver
    solver = RedCNNSolver(config=train_config, model=model,
                          eval_loader=eval_loader, eval_logger=eval_logger)
    solver.evaluate()