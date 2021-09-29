from collections import namedtuple
import torch
from torch import nn
import pdb
import os
import numpy as np
import csv 
import random
import librosa

split_ratio = 0.9
seed_num = 100
random.seed(seed_num)

correct_csv_file = '../correct.csv'
with open(correct_csv_file, newline='') as f:
    reader = csv.reader(f)
    tmp = list(reader)
    correct_list = [x[-1].split('/')[-1] for x in tmp]
f.close()

data_path = '../dataset'
csv_path = '../csv'
csv_list = [f for f in os.listdir(csv_path) if f.split('.')[-1] == 'csv']

total_list = []
for fi in csv_list:
    csv_file = os.path.join(csv_path,fi)
    #with open(csv_file, newline='',encoding='cp949') as f:
    with open(csv_file, newline='') as f:
        reader = csv.reader(f)
        tmp = list(reader)
        data_name_list = tmp[0]
        total_list += tmp[1:]
    f.close()

t_audio, v_audio = [], []
t_text, v_text = [], []
t_label, v_label = [], []
class_dict = {'negative': [],'neutral': [],'positive': []}
data_label_idx = data_name_list.index('mul_emotion')
class_dict['negative'] = [(x,'negative') for x in total_list if x[data_label_idx] != 'happy' and x[data_label_idx] != 'neutral' and x[data_label_idx] != 'surprise' and x[0].split('/')[-1] in correct_list]
class_dict['neutral'] = [(x,'neutral') for x in total_list if x[data_label_idx] == 'neutral' and x[0].split('/')[-1] in correct_list]
class_dict['positive'] = [(x,'positive') for x in total_list if x[data_label_idx] == 'happy' and x[0].split('/')[-1] in correct_list]


train_list,val_list = [],[]
data_num = 7113 
for class_name in class_dict.keys():
    temp_list = class_dict[class_name]
    random.shuffle(temp_list)
    if len(temp_list) < data_num:
        split_idx = int(split_ratio*len(temp_list))
        train_list += temp_list[:split_idx]
        val_list += temp_list[split_idx:len(temp_list)]
    else:
        split_idx = int(split_ratio*data_num)
        train_list += temp_list[:split_idx]
        val_list += temp_list[split_idx:data_num]


'''Train Data'''
for data in train_list:
    t_label.append(data[1].replace("\ufeff","")) 
    t_audio.append(os.path.join(data_path,data[0][0]))
    t_text.append(data[0][1])

'''Valid Data'''
for data in val_list:
    v_label.append(data[1].replace("\ufeff","")) 
    v_audio.append(os.path.join(data_path,data[0][0]))
    v_text.append(data[0][1])


count_dict = {}
count_dict["over_twenty"] = 0
count_dict["over_fifteen"] = 0
count_dict["over_ten"] = 0
count_dict["over_five"] = 0
count_dict["below_five"] = 0

pdb.set_trace()
for audio_path in t_audio:
    audio, _ = librosa.load(audio_path, sr=16000, dtype=np.float32)
    audio_len = len(audio) / 16000
    print("{}: {}".format(audio_path,audio_len))
    if audio_len > 20:
        count_dict["over_twenty"] +=1
    elif audio_len > 15:   
        count_dict["over_fifteen"] +=1
    elif audio_len > 10:   
        count_dict["over_ten"] +=1
    elif audio_len > 5:
        count_dict["over_five"] +=1
    else:
        count_dict["below_five"] +=1

for audio_path in v_audio:
    audio, _ = librosa.load(audio_path, sr=16000, dtype=np.float32)
    audio_len = len(audio) / 16000
    print("{}: {}".format(audio_path,audio_len))
    if audio_len > 20:
        count_dict["over_twenty"] +=1
    elif audio_len > 15:   
        count_dict["over_fifteen"] +=1
    elif audio_len > 10:   
        count_dict["over_ten"] +=1
    elif audio_len > 5:
        count_dict["over_five"] +=1
    else:
        count_dict["below_five"] +=1


for class_name in count_dict.keys():
    print('{}: {}'.format(class_name, count_dict[class_name]))