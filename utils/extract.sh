#!/bin/bash

# Data directory
DATASET_DIR='/home/nas/DB/AI_grand_challenge_2020/2020/t3_audio/'

# Feature directory
FEATURE_DIR='/home/minseok/Audio/AI_Challenge/features'

# Workspace
WORKSPACE='/home/minseok/Audio/AI_Challenge'
cd $WORKSPACE

########### Hyper-parameters ###########
FEATURE_TYPE='logmelgccintensity'  # 'logmel' | 'logmelgcc' | 'logmelintensity' | 'logmelgccintensity'
AUDIO_TYPE='mic'                # 'mic' | 'foa' | 'foa&mic'

############ Extract Features ############
# dev
python utils/feature_extractor.py --dataset_dir=$DATASET_DIR --feature_dir=$FEATURE_DIR --feature_type=$FEATURE_TYPE --data_type='dev' --audio_type=$AUDIO_TYPE

# eval
python utils/feature_extractor.py --dataset_dir=$DATASET_DIR --feature_dir=$FEATURE_DIR --feature_type=$FEATURE_TYPE --data_type='eval' --audio_type=$AUDIO_TYPE



