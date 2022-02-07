# Sound-Source-Localization
Sound Source Localization study for AI Grand Challenge 2021 (sponsored by NC Soft Vision Lab)

## Preparation 
### 1. Create the environment.     
```
$ cd Sound-Source-Localization/
$ conda create -y -n varco python=3.8
$ conda activate varco

####select according to your conda version: https://pytorch.org/####
$ conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia

$ conda install -y pandas h5py scipy
$ conda install -y pysoundfile librosa youtube-dl tqdm -c conda-forge
$ pip install PyYAML
$ pip install tensorboard
```     

### 2. Place the data-listed csv file in the path below.
```
Sound-Source-Localization/    
└── dataset/
    └── dataset.csv
```   
       
### 3. Run main.sh for training   
```
$ cd Sound-Source-Localization/
$ sh train.sh
```       
            
### 4. Run test.sh for test   
```
$ cd Sound-Source-Localization/
$ sh test.sh
```   

## Acknowledgement 
이 데이터는 2021년도 정부(과학기술정보통신부)의 재원으로 정보통신기획평가원의 지원을 받아 수행된 연구의 결과물임 (No.171125972, 인명 구조용 드론을 위한 영상/음성 인지 기술 개발)

This work was supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government(MSIT) (No.171125972, Audio-Visual Perception for Autonomous Rescue Drones)

