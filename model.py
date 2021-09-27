import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from utils.utils import make_encoder

import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model_utilities import ConvBlock, init_gru, init_layer, interpolate


class CRNN9(nn.Module):
    def __init__(self, class_num, pool_type='avg', pool_size=(2,2), pretrained_path=None):
        
        super().__init__()
        self.class_num = class_num
        self.pool_type = pool_type
        self.pool_size = pool_size
        self.interp_ratio = 8
        
        self.conv_block1 = ConvBlock(in_channels=3, out_channels=128)    # 1: 7, 128     2: 7, 64
        self.conv_block2 = ConvBlock(in_channels=128, out_channels=256)  # 1: 128, 256   2: 64, 256
        self.conv_block3 = ConvBlock(in_channels=256, out_channels=512)

        #self.gru = nn.GRU(input_size=512, hidden_size=256, 
        #    num_layers=2, dropout=0.3, batch_first=True, bidirectional=True)

        self.azimuth_fc = nn.Linear(512, class_num, bias=True)


        self.init_weights()

    def init_weights(self):

        #init_gru(self.gru)
        init_layer(self.azimuth_fc)


    def forward(self, x):
        '''input: (batch_size, mic_channels, time_steps, mel_bins)'''
        
        x = self.conv_block1(x, self.pool_type, pool_size=self.pool_size)
        x = self.conv_block2(x, self.pool_type, pool_size=self.pool_size)
        x = self.conv_block3(x, self.pool_type, pool_size=self.pool_size)
        '''(batch_size, feature_maps, time_steps, mel_bins)'''

        if self.pool_type == 'avg':
            x = torch.mean(x, dim=3)
        elif self.pool_type == 'max':
            (x, _) = torch.max(x, dim=3)
        '''(batch_size, feature_maps, time_steps)'''

        x = x.transpose(1,2)
        ''' (batch_size, time_steps, feature_maps):'''
        # 
        # self.gru.flatten_parameters()
        
        '''if pack padded'''
        # '''else'''
        #(x, _) = self.gru(x)

        azimuth_output = self.azimuth_fc(x)
        # Interpolate
        output = interpolate(azimuth_output, self.interp_ratio) 

        return output


class pretrained_CRNN8(CRNN9):

    def __init__(self, class_num, pool_type='avg', pool_size=(2,2), pretrained_path=None):

        super().__init__(class_num, pool_type, pool_size, pretrained_path=pretrained_path)
        if pretrained_path:
            self.load_weights(pretrained_path)

        self.gru = nn.GRU(input_size=512, hidden_size=256, 
            num_layers=1, batch_first=True, bidirectional=True)

        init_gru(self.gru)
        init_layer(self.azimuth_fc)

    def load_weights(self, pretrained_path):

        model = CRNN9(self.class_num, self.pool_type, self.pool_size)
        checkpoint = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['model_state_dict'])

        self.conv_block1 = model.conv_block1
        self.conv_block2 = model.conv_block2
        self.conv_block3 = model.conv_block3



class CRNN11(nn.Module):
    def __init__(self, class_num, pool_type='avg', pool_size=(2,2), pretrained_path=None):
        
        super().__init__()

        self.class_num = class_num
        self.pool_type = pool_type
        self.pool_size = pool_size
        self.interp_ratio = 16
        
        self.conv_block1 = ConvBlock(in_channels=3, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)

        self.gru = nn.GRU(input_size=512, hidden_size=256, 
            num_layers=2, dropout=0.3, batch_first=True, bidirectional=True)

        self.azimuth_fc = nn.Linear(512, class_num, bias=True)
        self.init_weights()

    def init_weights(self):

        init_gru(self.gru)
        init_layer(self.azimuth_fc)

    def forward(self, x):
        '''input: (batch_size, mic_channels, time_steps, mel_bins)'''

        x = self.conv_block1(x, self.pool_type, pool_size=self.pool_size)
        x = self.conv_block2(x, self.pool_type, pool_size=self.pool_size)
        x = self.conv_block3(x, self.pool_type, pool_size=self.pool_size)
        x = self.conv_block4(x, self.pool_type, pool_size=self.pool_size)
        '''(batch_size, feature_maps, time_steps, mel_bins)'''

        if self.pool_type == 'avg':
            x = torch.mean(x, dim=3)
        elif self.pool_type == 'max':
            (x, _) = torch.max(x, dim=3)
        '''(batch_size, feature_maps, time_steps)'''

        x = x.transpose(1,2)
        ''' (batch_size, time_steps, feature_maps):'''

        # self.gru.flatten_parameters()
        (x, _) = self.gru(x)
        
        azimuth_output = self.azimuth_fc(x)
        '''(batch_size, time_steps, class_num)'''

        # Interpolate
        azimuth_output = interpolate(azimuth_output, self.interp_ratio) 
        return azimuth_output


class pretrained_CRNN10(CRNN11):

    def __init__(self, class_num, pool_type='avg', pool_size=(2,2), pretrained_path=None):

        super().__init__(class_num, pool_type, pool_size, pretrained_path=pretrained_path)
        
        if pretrained_path:
            self.load_weights(pretrained_path)

        self.gru = nn.GRU(input_size=512, hidden_size=256, 
            num_layers=1, batch_first=True, bidirectional=True)

        init_gru(self.gru)
        init_layer(self.azimuth_fc)

    def load_weights(self, pretrained_path):

        model = CRNN11(self.class_num, self.pool_type, self.pool_size)
        checkpoint = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['model_state_dict'])

        self.conv_block1 = model.conv_block1
        self.conv_block2 = model.conv_block2
        self.conv_block3 = model.conv_block3
        self.conv_block4 = model.conv_block4



class Gated_CRNN9(nn.Module):
    def __init__(self, class_num, pool_type='avg', pool_size=(2,2), pretrained_path=None):
        
        super().__init__()

        self.class_num = class_num
        self.pool_type = pool_type
        self.pool_size = pool_size
        self.interp_ratio = 8
        
        self.conv_block1 = ConvBlock(in_channels=3, out_channels=64)    # 1: 7, 128     2: 7, 64
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=256)  # 1: 128, 256   2: 64, 256
        self.conv_block3 = ConvBlock(in_channels=256, out_channels=512)

        self.gate_block1 = ConvBlock(in_channels=3, out_channels=64)
        self.gate_block2 = ConvBlock(in_channels=64, out_channels=256)
        self.gate_block3 = ConvBlock(in_channels=256, out_channels=512)

        self.gru = nn.GRU(input_size=512, hidden_size=256, 
            num_layers=2, dropout=0.3, batch_first=True, bidirectional=True)

        #self.azimuth_fc = nn.Linear(512, class_num, bias=True)
        self.azimuth_fc1 = nn.Linear(512, 128, bias=True)
        self.azimuth_fc2 = nn.Linear(128, class_num, bias=True)

        self.init_weights()

    def init_weights(self):

        init_gru(self.gru)
        #init_layer(self.azimuth_fc)
        init_layer(self.azimuth_fc1)
        init_layer(self.azimuth_fc2)

    def forward(self, x):
        #pdb.set_trace() 
        '''input: (batch_size, mic_channels, time_steps, mel_bins)'''
        gate = self.gate_block1(x, self.pool_type, pool_size=self.pool_size)
        x = self.conv_block1(x, self.pool_type, pool_size=self.pool_size)
        x = x * torch.sigmoid(gate)

        gate = self.gate_block2(x, self.pool_type, pool_size=self.pool_size)
        x = self.conv_block2(x, self.pool_type, pool_size=self.pool_size)
        x = x * torch.sigmoid(gate)

        gate = self.gate_block3(x, self.pool_type, pool_size=self.pool_size)
        x = self.conv_block3(x, self.pool_type, pool_size=self.pool_size)
        x = x * torch.sigmoid(gate)
        '''(batch_size, feature_maps, time_steps, mel_bins)'''

        if self.pool_type == 'avg':
            x = torch.mean(x, dim=3)
        elif self.pool_type == 'max':
            (x, _) = torch.max(x, dim=3)
        '''(batch_size, feature_maps, time_steps)'''
        
        x = x.transpose(1,2)
        ''' (batch_size, time_steps, feature_maps):'''

        self.gru.flatten_parameters()
        (x, _) = self.gru(x)
        
        x = self.azimuth_fc1(x)
        azimuth_output = self.azimuth_fc2(x)
        #azimuth_output = self.azimuth_fc(x)
        '''(batch_size, time_steps, class_num)'''

        # Interpolate
        azimuth_output = interpolate(azimuth_output, self.interp_ratio)
        azimuth_output = F.sigmoid(azimuth_output)
        
        return azimuth_output


class pretrained_Gated_CRNN8(Gated_CRNN9):

    def __init__(self, class_num, pool_type='avg', pool_size=(2,2), pretrained_path=None):

        super().__init__(class_num, pool_type, pool_size, pretrained_path=pretrained_path)
        
        if pretrained_path:
            self.load_weights(pretrained_path)

        self.gru = nn.GRU(input_size=512, hidden_size=256, 
            num_layers=1, batch_first=True, bidirectional=True)

        init_gru(self.gru)
        #init_layer(self.azimuth_fc)
        init_layer(self.azimuth_fc1)
        init_layer(self.azimuth_fc2)

    def load_weights(self, pretrained_path):

        model = Gated_CRNN9(self.class_num, self.pool_type, self.pool_size)
        checkpoint = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['model_state_dict'])

        self.conv_block1 = model.conv_block1
        self.conv_block2 = model.conv_block2
        self.conv_block3 = model.conv_block3

class JUNGMIN(nn.Module):
    def __init__(self):
        super().__init__()

        in_channel = 514
        out_channel = 514

        self.layer1 = nn.Sequential(
                        nn.Conv1d(in_channel, in_channel, kernel_size=5,stride=1,padding=2,groups=in_channel),
                        nn.Conv1d(in_channel, out_channel,kernel_size=1,stride=1,padding=0),
                        nn.BatchNorm1d(out_channel),
                        nn.ReLU(),
        )



        self.layer2 = nn.Sequential(
                        nn.Conv1d(in_channel, in_channel, kernel_size=5,stride=1,padding=2,groups=in_channel),
                        nn.Conv1d(in_channel, out_channel,kernel_size=1,stride=1,padding=0),
                        nn.BatchNorm1d(out_channel),
                        nn.ReLU(),
        )

        self.layer3 = nn.Sequential(
                        nn.Conv1d(in_channel, in_channel, kernel_size=5,stride=1,padding=2,groups=in_channel),
                        nn.Conv1d(in_channel, out_channel,kernel_size=1,stride=1,padding=0),
                        nn.BatchNorm1d(out_channel),
                        nn.ReLU(),
        )
        
        self.fc = nn.Linear(514, 10)
        self.init_weights()

    def init_weights(self):
        init_layer(self.fc)

    def forward(self, x):
        
        B,M,T,F = x.size()
        # pdb.set_trace()
        '''input: (batch_size, mic_channels, time_steps, mel_bins)'''
        x =  x.permute(0, 1, 3, 2).reshape(B, -1, T)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.transpose(1,2)
        
        # (x, _) = self.gru(x)
        return self.fc(x)

class JUNGMIN2(nn.Module):
    def __init__(self):
        super().__init__()

        in_channel = 514
        out_channel = 514

        self.layer1 = nn.Sequential(
                        nn.Conv1d(in_channel, in_channel, kernel_size=5,stride=1,padding=2,groups=in_channel),
                        nn.Conv1d(in_channel, out_channel,kernel_size=1,stride=1,padding=0),
                        nn.BatchNorm1d(out_channel),
                        nn.ReLU(),
        )



        self.layer2 = nn.Sequential(
                        nn.Conv1d(in_channel, in_channel, kernel_size=5,stride=1,padding=2,groups=in_channel),
                        nn.Conv1d(in_channel, out_channel,kernel_size=1,stride=1,padding=0),
                        nn.BatchNorm1d(out_channel),
                        nn.ReLU(),
        )

        self.layer3 = nn.Sequential(
                        nn.Conv1d(in_channel, in_channel, kernel_size=5,stride=1,padding=2,groups=in_channel),
                        nn.Conv1d(in_channel, out_channel,kernel_size=1,stride=1,padding=0),
                        nn.BatchNorm1d(out_channel),
                        nn.ReLU(),
        )
        
        
        self.gru = nn.GRU(input_size=514, hidden_size=256, 
            num_layers=2, dropout=0.3, batch_first=True, bidirectional=True)

        self.fc = nn.Linear(512, 10)

        self.init_weights()

    def init_weights(self):

        init_gru(self.gru)
        init_layer(self.fc)

    def forward(self, x):
        
        B,M,T,F = x.size()
        # pdb.set_trace()
        '''input: (batch_size, mic_channels, time_steps, mel_bins)'''
        x =  x.permute(0, 1, 3, 2).reshape(B, -1, T)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.transpose(1,2)
        
        (x, _) = self.gru(x)
        return self.fc(x)


class JUNGMIN3(nn.Module):
    def __init__(self):
        super().__init__()

        in_channel = 514
        out_channel = 514

        self.layer1 = nn.Sequential(
                        nn.Conv1d(in_channel, in_channel, kernel_size=5,stride=1,padding=2,groups=in_channel),
                        nn.Conv1d(in_channel, out_channel,kernel_size=1,stride=1,padding=0),
                        nn.BatchNorm1d(out_channel),
                        nn.ReLU(),
        )



        self.layer2 = nn.Sequential(
                        nn.Conv1d(in_channel, in_channel, kernel_size=5,stride=1,padding=2,groups=in_channel),
                        nn.Conv1d(in_channel, out_channel,kernel_size=1,stride=1,padding=0),
                        nn.BatchNorm1d(out_channel),
                        nn.ReLU(),
        )

        self.layer3 = nn.Sequential(
                        nn.Conv1d(in_channel, in_channel, kernel_size=5,stride=1,padding=2,groups=in_channel),
                        nn.Conv1d(in_channel, out_channel,kernel_size=1,stride=1,padding=0),
                        nn.BatchNorm1d(out_channel),
                        nn.ReLU(),
        )
        
        self.fc = nn.Sequential(
                    nn.Linear(514, 256),
                    nn.ReLU(),
                    nn.Linear(256, 10)
        )


        self.init_weights()

    def init_weights(self):
        init_layer(self.fc)

    def forward(self, x):
        
        B,M,T,F = x.size()
        # pdb.set_trace()
        '''input: (batch_size, mic_channels, time_steps, mel_bins)'''
        x =  x.permute(0, 1, 3, 2).reshape(B, -1, T)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.transpose(1,2)
        
        # (x, _) = self.gru(x)
        return self.fc(x)





class JUNGMIN4(nn.Module):
    def __init__(self):
        super().__init__()

        in_channel = 514
        out_channel = 514

        self.layer1 = nn.Sequential(
                        nn.Conv1d(in_channel, in_channel, kernel_size=5,stride=1,padding=2,groups=in_channel),
                        nn.Conv1d(in_channel, out_channel,kernel_size=1,stride=1,padding=0),
                        nn.BatchNorm1d(out_channel),
                        nn.ReLU(),
        )



        self.layer2 = nn.Sequential(
                        nn.Conv1d(in_channel, in_channel, kernel_size=5,stride=1,padding=2,groups=in_channel),
                        nn.Conv1d(in_channel, out_channel,kernel_size=1,stride=1,padding=0),
                        nn.BatchNorm1d(out_channel),
                        nn.ReLU(),
        )

        self.layer3 = nn.Sequential(
                        nn.Conv1d(in_channel, in_channel, kernel_size=5,stride=1,padding=2,groups=in_channel),
                        nn.Conv1d(in_channel, out_channel,kernel_size=1,stride=1,padding=0),
                        nn.BatchNorm1d(out_channel),
                        nn.ReLU(),
        )
        
        
        self.layer4 = nn.Sequential(
                        nn.Conv1d(in_channel, in_channel, kernel_size=5,stride=1,padding=2,groups=in_channel),
                        nn.Conv1d(in_channel, out_channel,kernel_size=1,stride=1,padding=0),
                        nn.BatchNorm1d(out_channel),
                        nn.ReLU(),
        )
        


        self.fc = nn.Sequential(
                    nn.Linear(514, 256),
                    nn.ReLU(),
                    nn.Linear(256, 10)
        )


        self.init_weights()

    def init_weights(self):
        init_layer(self.fc)

    def forward(self, x):
        
        B,M,T,F = x.size()
        # pdb.set_trace()
        '''input: (batch_size, mic_channels, time_steps, mel_bins)'''
        x =  x.permute(0, 1, 3, 2).reshape(B, -1, T)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.transpose(1,2)
        
        # (x, _) = self.gru(x)
        return self.fc(x)




class JUNGMIN5(nn.Module):
    def __init__(self):
        super().__init__()

        in_channel = 514
        out_channel = 514

        self.layer1 = nn.Sequential(
                        nn.Conv1d(in_channel, in_channel, kernel_size=5,stride=1,padding=2,groups=in_channel),
                        nn.Conv1d(in_channel, out_channel,kernel_size=1,stride=1,padding=0),
                        nn.BatchNorm1d(out_channel),
                        nn.ReLU(),
        )



        self.layer2 = nn.Sequential(
                        nn.Conv1d(in_channel, in_channel, kernel_size=5,stride=1,padding=2,groups=in_channel),
                        nn.Conv1d(in_channel, out_channel,kernel_size=1,stride=1,padding=0),
                        nn.BatchNorm1d(out_channel),
                        nn.ReLU(),
        )

        self.layer3 = nn.Sequential(
                        nn.Conv1d(in_channel, in_channel, kernel_size=5,stride=1,padding=2,groups=in_channel),
                        nn.Conv1d(in_channel, out_channel,kernel_size=1,stride=1,padding=0),
                        nn.BatchNorm1d(out_channel),
                        nn.ReLU(),
        )
        
        
        self.layer4 = nn.Sequential(
                        nn.Conv1d(in_channel, in_channel, kernel_size=5,stride=1,padding=2,groups=in_channel),
                        nn.Conv1d(in_channel, out_channel,kernel_size=1,stride=1,padding=0),
                        nn.BatchNorm1d(out_channel),
                        nn.ReLU(),
        )
        


        self.fc = nn.Sequential(
                    nn.Linear(514, 256),
                    nn.ReLU(),
                    nn.Linear(256, 10)
        )


        self.init_weights()

    def init_weights(self):
        init_layer(self.fc)

    def forward(self, x):
        
        B,M,T,F = x.size()
        # pdb.set_trace()
        '''input: (batch_size, mic_channels, time_steps, mel_bins)'''
        x =  x.permute(0, 1, 3, 2).reshape(B, -1, T)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.transpose(1,2)
        
        # (x, _) = self.gru(x)
        return self.fc(x)
