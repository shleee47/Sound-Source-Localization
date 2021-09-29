import os
import sys
import time
import numpy as np
import datetime

import pickle as pkl

from pathlib import Path
import torch
import pdb
from tqdm import tqdm
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn.functional as F

import logging
import json
from multiprocessing import Pool
import time
import warnings
warnings.filterwarnings("ignore")

class ModelTrainer:

    def __init__(self, model, train_loader, valid_loader, criterion, optimizer, scheduler, config, epochs, device, save_path, ckpt_path=None, comment=None, fold=2):

        self.device = torch.device('cuda:{}'.format(device))
        #self.model = model.to(self.device)
        self.model = model.cuda()

        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.exp_path = Path(os.path.join(save_path, datetime.now().strftime('%d%B_%0l%0M'))) #21November_0430
        self.exp_path.mkdir(exist_ok=True, parents=True)

        # Set logger
        self.logger = logging.getLogger('')
        self.logger.setLevel(logging.INFO)
        fh = logging.FileHandler(os.path.join(self.exp_path, 'training.log'))
        sh = logging.StreamHandler(sys.stdout)
        self.logger.addHandler(fh)
        self.logger.addHandler(sh)
        
        #Dump hyper-parameters
        with open(str(self.exp_path.joinpath('config.json')), 'w') as f:
            json.dump(config, f, indent=2)

        if comment != None:
            self.logger.info(comment)

        self.writter = SummaryWriter(self.exp_path.joinpath('logs'))
        self.epochs = epochs
        self.best_acc = 0.0
        self.best_epoch = 0
        
        if ckpt_path != None:
            self.load_checkpoint(ckpt_path)
            self.optimizer.param_groups[0]['lr'] = 0.0001

    def train(self):
        for epoch in tqdm(range(self.epochs)):
            start = time.time()
            train_loss, t_accuracy= self.train_single_epoch(epoch)
            valid_loss, v_accuracy = self.inference()
            duration = time.time() - start

            if v_accuracy > self.best_acc:
                self.best_acc = v_accuracy
                self.best_epoch = epoch

            self.scheduler.step(v_accuracy)
            self.logger.info("epoch: {} --- t_loss : {:0.3f}, train_acc = {}%, v_loss: {:0.3f}, val_acc: {}%, best_acc: {}%, best_epoch: {}, time: {:0.2f}s, lr: {}"\
                                                            .format(epoch, train_loss, t_accuracy, valid_loss, v_accuracy, self.best_acc, self.best_epoch, duration,self.optimizer.param_groups[0]['lr']))
    
            self.save_checkpoint(epoch, v_accuracy)

            self.writter.add_scalar('data/Train_Loss', train_loss, epoch)
            self.writter.add_scalar('data/Valid_Loss', valid_loss, epoch)
            self.writter.add_scalar('data/Train_Accuracy', t_accuracy, epoch)
            self.writter.add_scalar('data/Valid_Accuracy', v_accuracy, epoch)

        self.writter.close()


    def train_single_epoch(self, epoch):
        self.model.train()
        
        total_loss = 0.0
        accuracy = 0.0
        correct_cnt = 0
        tot_cnt = 0
        batch_size = len(self.train_loader)

        for b, batch in (enumerate(self.train_loader)):

            inputs, labels = batch
            B, C, T, Freq = inputs.size()
            inputs = inputs.cuda()
            labels = labels.cuda()

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            scores = outputs.mean(1)
            best_prediction = scores.max(-1)[1]

            for i in range(B):
                if labels[i, best_prediction[i]] == 1.0:
                    correct_cnt += 1
            
            batch_loss = self.criterion(scores, labels)
            batch_loss.backward()
            total_loss += batch_loss.item()
            self.optimizer.step()
            tot_cnt += B

            print("{}/{}: {}/{}".format(b, batch_size, correct_cnt, tot_cnt), end='\r')

        mean_loss = total_loss / tot_cnt
        return mean_loss, (correct_cnt/tot_cnt)*100


    def inference(self):
        self.model.eval()
        
        total_loss = 0.0
        accuracy = 0.0
        correct_cnt = 0
        tot_cnt = 0
        batch_size = len(self.valid_loader)
        with torch.no_grad():
            for b, batch in enumerate(self.valid_loader):

                inputs, labels = batch
                B, C, T, Freq = inputs.size()  
                inputs = inputs.cuda()
                labels = labels.cuda()
                outputs = self.model(inputs)

                scores = outputs.mean(1)
                best_prediction = scores.max(-1)[1]

                for i in range(B):
                    if labels[i, best_prediction[i]] == 1.0:
                        correct_cnt += 1

                batch_loss = self.criterion(scores, labels)
                total_loss += batch_loss.item()
                tot_cnt += B

                print("{}/{}: {}/{}".format(b, batch_size, correct_cnt, tot_cnt), end='\r')

        mean_loss = total_loss / tot_cnt
        return mean_loss, (correct_cnt/tot_cnt)*100


    def load_checkpoint(self, ckpt):
        self.logger.info("Loading checkpoint from {ckpt}")
        print('Loading checkpoint : {}'.format(ckpt))
        checkpoint = torch.load(ckpt)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.optimizer.load_state_dict(checkpoint['optimizer'])#, strict=False)


    def save_checkpoint(self, epoch, vacc, best=True):
        
        state_dict = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        
        self.exp_path.joinpath('ckpt').mkdir(exist_ok=True, parents=True)
        save_path = "{}/ckpt/{}_{:0.4f}.pt".format(self.exp_path, epoch, vacc)
        torch.save(state_dict, save_path)


class ModelTester:
    def __init__(self, model, test_loader, ckpt_path, device):

        # Essential parts
        self.device = torch.device('cuda:{}'.format(device))
        #self.model = model.to(self.device)
        self.model = model.cuda()
        self.test_loader = test_loader
        # Set logger
        self.logger = logging.getLogger('')
        self.logger.setLevel(logging.INFO)
        sh = logging.StreamHandler(sys.stdout)
        self.logger.addHandler(sh)

        self.load_checkpoint(ckpt_path)


    def load_checkpoint(self, ckpt):
        self.logger.info(f"Loading checkpoint from {ckpt}")
        # print('Loading checkpoint : {}'.format(ckpt))
        checkpoint = torch.load(ckpt)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        


    def test(self):
        """
        images : [B x T x C x H x W]
        labels : [B x T]
        """
        self.model.eval()
        result = ['FA','MA']
        batch_size = len(self.test_loader)
        final = open('/home/ygchoi/gender_detection/result.csv', 'w')
        final.write('filename'+'\t'+'prediction'+'\n')

        with torch.no_grad():
            for b, batch in tqdm(enumerate(self.test_loader), total=len(self.test_loader)):
                
                inputs, audio_path = batch
                inputs = torch.unsqueeze(inputs,1)
                B, C, T, Freq = inputs.size()  
                inputs = inputs.cuda()
                outputs = self.model(inputs)
                best_prediction = outputs.max(2)[1].mode()[0]
                final.write(audio_path[0]+'\t'+result[best_prediction.item()]+'\n')
        
        final.close()
