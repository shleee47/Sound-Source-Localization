conda create -y -n vamos python=3.8
conda activate vamos
#source activate vamos
conda install scipy

####select according to your conda version####
####https://pytorch.org/####
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
#conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch

conda install pyaudio
conda install -c conda-forge librosa

pip install PyYAML
