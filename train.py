# train.py
import os
from networks import get_model
from solver import get_solver
from utils import get_loader, get_logger, get_transforms

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
    train_loader_config = train_config["train"]["data_loader"]
    train_loader = get_loader(mode='train',
                              data_path=train_config['data_path'],
                              transform=get_transforms(config=train_loader_config.get('transform', {})),
                              batch_size=train_loader_config.get('batch_size', 16),
                              num_workers=train_loader_config.get('num_workers', 0),
                              shuffle=train_loader_config.get('shuffle', True))

    val_loader_config = train_config['validate']['data_loader']
    val_loader = get_loader(mode='val',
                            data_path=train_config['data_path'],
                            transform=get_transforms(config=val_loader_config.get('transform', {})),
                            batch_size=val_loader_config.get('batch_size', 16),
                            num_workers=val_loader_config.get('num_workers', 0),
                            shuffle=val_loader_config.get('shuffle', True))
    

    # Build model
    ModelClass = get_model(train_config['network']['type'])
    model = ModelClass()

    # Solver
    Solver = get_solver(train_config["workframe"])
    solver = Solver(config=train_config, model=model,
                          train_loader=train_loader, train_logger=train_logger,
                          val_loader=val_loader, val_logger=val_logger)
    solver.train()