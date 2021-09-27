from torch import optim
from torch.optim import lr_scheduler
import torch

def setup_solver(parameters, config):
    if config['optimizer']['name'] == 'Adam':
        optimizer = optim.Adam(parameters, lr=config['optimizer']['lr'], weight_decay=config['optimizer']['weight_decay'],
                                eps=config['optimizer']['eps'])
        #                       eps=1e-3 if args.fp16 else 1e-8)
    elif config['optimizer']['name'] == 'SGD':
        optimizer = optim.SGD(parameters, lr=config['optimizer']['lr'], weight_decay=config['optimizer']['weight_decay'],
                              momentum=0.9, nesterov=True)


    if config['scheduler']['name'] == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=config['scheduler']['patience'], threshold=0.01, cooldown=0,
                                                   threshold_mode='abs', mode='max', factor=config['scheduler']['factor'])


    elif config['scheduler']['name'] == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9, last_epoch=-1)

    elif config['scheduler']['name'] == 'exp':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.99)



    elif config['scheduler']['name'] == 'cycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=0.001, 
            total_steps=10000,
            pct_start=0.3,
            base_momentum=0.9*0.95,
            max_momentum=0.95,
            final_div_factor=1/0.0001,
        )

    # else:  # args.scheduler == 'step':
    #     scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.gamma)

    return optimizer, scheduler

