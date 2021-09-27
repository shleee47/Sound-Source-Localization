import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append('./')
import argparse
import torch
import torch.nn as nn
import pdb
import yaml 
import numpy as np
from torch.utils.data import DataLoader
import os
import pickle
from pathlib import Path
from trainer import ModelTrainer, ModelTester
from utils.setup import setup_solver
from utils.loss import create_criterion
from utils.utils import tr_val_split
from datasets import Audio_Reader, Audio_Collate, Test_Reader
from model import pretrained_Gated_CRNN8

def train(config):

    #pdb.set_trace()
    '''Dataset Preparation'''
    train_list, val_list = tr_val_split(config['datasets']['csv'])
    
    '''Data loader'''
    train_dataset = Audio_Reader(train_list)
    train_loader = DataLoader(dataset=train_dataset, batch_size=config['dataloader']['train']['batch_size'], shuffle=True, collate_fn=lambda x: Audio_Collate(x), num_workers=config['dataloader']['train']['num_workers'])
    valid_dataset = Audio_Reader(val_list)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=config['dataloader']['valid']['batch_size'], shuffle=True, collate_fn=lambda x: Audio_Collate(x), num_workers=config['dataloader']['valid']['num_workers'])
    
    '''Model / Loss Criterion / Optimizer/ Scheduler'''
    SSL_model = pretrained_Gated_CRNN8(10)
    criterion = create_criterion(config['criterion']['name'])
    optimizer, scheduler = setup_solver(SSL_model.parameters(), config)

    '''Trainer'''
    trainer = ModelTrainer(SSL_model, train_loader, valid_loader, criterion, optimizer, scheduler, config, **config['trainer'])
    trainer.train()

def test(config):

    test_dataset = Test_Reader(config['datasets']['test'])
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, pin_memory = True, num_workers=0)

    SSL_model = pretrained_Gated_CRNN8(10)

    tester = ModelTester(SSL_model, test_loader, config['tester']['ckpt_path'], config['tester']['device'])
    tester.test()

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--base_dir', type=str, default='.', help='Root directory')
    parser.add_argument('-c', '--config', type=str, help='Path to option YAML file.')
    parser.add_argument('-d', '--dataset', type=str, help='Dataset')
    parser.add_argument('-m', '--mode', type=str, help='Train or Test')
    args = parser.parse_args()
    
    '''Load Config'''
    with open(os.path.join(args.config, args.dataset + '.yml'), mode='r') as f:
        config = yaml.load(f,Loader=yaml.FullLoader)

    if args.mode == 'Train':
        train(config)
    elif args.mode == 'Test':
        test(config)
