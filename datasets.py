import json
import sys
sys.path.append('..')
from torch.utils.data.dataset import Dataset
from pathlib import Path
import pickle
import pdb
import torch
import numpy as np
import argparse
import os
import sys
# import h5py
import librosa
import numpy as np
# import pandas as pd
import scipy.io as sio
from scipy import signal
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

class Audio_Reader(Dataset):
    def __init__(self, datalist):
        super(Audio_Reader, self).__init__()
        self.datalist = datalist
        self.classlist = ['0','20','40','60','80','100','120','140','160','180']
        self.nfft = 512
        self.hopsize = self.nfft // 4
        self.window = 'hann'

    def __len__(self):
        return len(self.datalist)

    def LogMelGccExtractor(self, sig):
        def logmel(sig):

            #pdb.set_trace()
            S = np.abs(librosa.stft(y=sig,
                                    n_fft=self.nfft,
                                    hop_length=self.hopsize,
                                    center=True,
                                    window=self.window,
                                    pad_mode='reflect'))**2        
        
            # S_mel = np.dot(self.melW, S).T
            S = librosa.power_to_db(S**2, ref=1.0, amin=1e-10, top_db=None)
            S = np.expand_dims(S, axis=0)

            return S

        def gcc_phat(sig, refsig):

            #pdb.set_trace()
            Px = librosa.stft(y=sig,
                            n_fft=self.nfft,
                            hop_length=self.hopsize,
                            center=True,
                            window=self.window, 
                            pad_mode='reflect')

            Px_ref = librosa.stft(y=refsig,
                                n_fft=self.nfft,
                                hop_length=self.hopsize,
                                center=True,
                                window=self.window,
                                pad_mode='reflect')
        
            R = Px*np.conj(Px_ref)
            return R

        def transform(audio):

            channel_num = audio.shape[0]
            feature_logmel = []
            feature_gcc_phat = []
            for n in range(channel_num):
                feature_logmel.append(logmel(audio[n]))
                for m in range(n+1, channel_num):
                    feature_gcc_phat.append(
                        gcc_phat(sig=audio[m], refsig=audio[n]))
            
            #pdb.set_trace()
            feature_logmel = np.concatenate(feature_logmel, axis=0)
            feature_gcc_phat = np.concatenate(feature_gcc_phat, axis=0)
            feature = np.concatenate([feature_logmel, np.expand_dims(feature_gcc_phat, axis=0)])

            return feature
        
        return transform(sig)

    def __getitem__(self, idx):

        audio_path = self.datalist[idx]
        class_name = audio_path.split('/')[-2].strip('degree') 
        class_num = self.classlist.index(class_name)
        audio, _ = librosa.load(audio_path, sr=16000, mono=False, dtype=np.float32)
        if audio.shape[1] >80000:
            audio = audio[:,:80000]

        feature = self.LogMelGccExtractor(audio)
        #pdb.set_trace()
        return torch.FloatTensor(feature).transpose(1,2), np.array([class_num])


def Audio_Collate(batch):
   
    #pdb.set_trace()
    data, class_num = list(zip(*batch))
    data_len = torch.LongTensor(np.array([x.size(1) for x in data if x.size(1)!=1]))
    #if len(data_len) == 0:
    #    return -1

    max_len = max(data_len)
    wrong_indices = []
    
    #for i, a_ in enumerate(class_num):
    #    if a_[0] == -1:
    #        wrong_indices.append(i)

    B = len(data)
    #pdb.set_trace()
    #inputs = torch.zeros(B-len(wrong_indices), 1, max_len, 10)
    #labels = torch.zeros(B-len(wrong_indices), 2)
    inputs = torch.zeros(B, 3, max_len, 257)
    labels = torch.zeros(B, 10)
    j = 0
    #pdb.set_trace()
    '''zero pad'''    
    for i in range(B):
        #if i in wrong_indices:
        #    continue

        inputs[j, : , :data[i].size(1),:] = data[i]
        labels[j, class_num[i]] = 1.0
        j += 1

    #pdb.set_trace()
    #data = (inputs, labels, data_len)
    data = (inputs, labels)
    return data


class Test_Reader(Dataset):
    def __init__(self, datalist):
        super(Audio_Reader, self).__init__()
        self.datalist = datalist
        self.classlist = ['0','20','40','60','80','100','120','140','160','180']
        self.nfft = 512
        self.hopsize = self.nfft // 4
        self.window = 'hann'

    def __len__(self):
        return len(self.datalist)

    def LogMelGccExtractor(self, sig):
        def logmel(sig):

            #pdb.set_trace()
            S = np.abs(librosa.stft(y=sig,
                                    n_fft=self.nfft,
                                    hop_length=self.hopsize,
                                    center=True,
                                    window=self.window,
                                    pad_mode='reflect'))**2        
        
            # S_mel = np.dot(self.melW, S).T
            S = librosa.power_to_db(S**2, ref=1.0, amin=1e-10, top_db=None)
            S = np.expand_dims(S, axis=0)

            return S

        def gcc_phat(sig, refsig):

            #pdb.set_trace()
            Px = librosa.stft(y=sig,
                            n_fft=self.nfft,
                            hop_length=self.hopsize,
                            center=True,
                            window=self.window, 
                            pad_mode='reflect')

            Px_ref = librosa.stft(y=refsig,
                                n_fft=self.nfft,
                                hop_length=self.hopsize,
                                center=True,
                                window=self.window,
                                pad_mode='reflect')
        
            R = Px*np.conj(Px_ref)
            return R

        def transform(audio):

            channel_num = audio.shape[0]
            feature_logmel = []
            feature_gcc_phat = []
            for n in range(channel_num):
                feature_logmel.append(logmel(audio[n]))
                for m in range(n+1, channel_num):
                    feature_gcc_phat.append(
                        gcc_phat(sig=audio[m], refsig=audio[n]))
            
            #pdb.set_trace()
            feature_logmel = np.concatenate(feature_logmel, axis=0)
            feature_gcc_phat = np.concatenate(feature_gcc_phat, axis=0)
            feature = np.concatenate([feature_logmel, np.expand_dims(feature_gcc_phat, axis=0)])

            return feature
        
        return transform(sig)

    def __getitem__(self, idx):

        audio_path = self.datalist[idx]
        class_name = audio_path.split('/')[-2].strip('degree') 
        class_num = self.classlist.index(class_name)
        audio, _ = librosa.load(audio_path, sr=16000, mono=False, dtype=np.float32)
        if audio.shape[1] >80000:
            audio = audio[:,:80000]

        feature = self.LogMelGccExtractor(audio)
        #pdb.set_trace()
        return torch.FloatTensor(feature).transpose(1,2), np.array([class_num])
