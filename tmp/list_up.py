import glob
import pdb
import random

pdb.set_trace()
data_path = '/home/nas/DB/AI_grand_challenge_2020/Soundproof/cutted/'
data_path_2 = '/home/nas/DB/AI_grand_challenge_2020/NewSoundproof/cutted/'

data_list = glob.glob(data_path+'**/*.wav')
data_list_2 = glob.glob(data_path_2+'**/*.wav')
random.shuffle(data_list)

Soundproof_dict = {}

real_data_list = []
for data in data_list:
    class_name = data.split('/')[-2]
    
    if class_name not in Soundproof_dict.keys():
        Soundproof_dict[class_name] = 0
    
    if Soundproof_dict[class_name] == 127: 
        continue

    Soundproof_dict[class_name] += 1
    real_data_list.append(data)

real_data_list= sorted(real_data_list)

pdb.set_trace()


csv_path = '/home/sanghoon/SSL/dataset/dataset.csv'
f = open(csv_path,'w')

pdb.set_trace()
total_data_list = data_list_2 + real_data_list
for data in total_data_list:
    f.write(data+'\n')

f.close()

