conda create -y -n varco python=3.8
conda activate varco

####select according to your conda version####
####https://pytorch.org/####
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
#conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch

conda install -y pandas h5py scipy
conda install -y pysoundfile librosa youtube-dl tqdm -c conda-forge

pip install PyYAML
pip install tensorboard
