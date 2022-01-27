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
