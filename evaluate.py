# eval.py
import os
from networks import get_model
from solver import get_solver
from utils import get_loader, get_logger, get_transforms

import yaml

def evaluate(args: dict):
    with open(f'config/{args.workframe}.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # args + yaml
    args_dict = vars(args)  # Namespace → dict
    overlap = set(args_dict).intersection(config)
    if overlap:
        print(f"[Warning] These keys from args will overwrite config: {overlap}")

    eval_config = {**config, **args_dict}
    
    # Logger
    log_path = os.path.join(eval_config['save_path'], "log")
    eval_logger = get_logger(log_dir=log_path, name='eval')
    
    # Data loader
    eval_loader_config = eval_config["evaluate"]["data_loader"]
    eval_loader = get_loader(mode='eval',
                             data_path=eval_config['data_path'],
                             transform=get_transforms(config=eval_loader_config.get('transform', {})),
                             batch_size=eval_loader_config.get('batch_size', 4),
                             num_workers=eval_loader_config.get('num_workers', 0),
                             shuffle=eval_loader_config.get('shuffle', True))

    # Build model
    ModelClass = get_model(eval_config['network']['type'])
    model = ModelClass()

    # Solver
    Solver = get_solver(eval_config["workframe"])
    solver = Solver(config=eval_config, model=model,
                          eval_loader=eval_loader, eval_logger=eval_logger)
    solver.evaluate()