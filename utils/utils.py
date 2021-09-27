from collections import namedtuple
import torch
from torch import nn
from utils.resnet import resnet18, resnet50, resnet101
import pdb
import random
import os
import csv
Encoder = namedtuple('Encoder', ('model', 'features', 'features_shape'))



def tr_val_split(data_csv):
    split_ratio = 0.9
    seed_num = 100
    random.seed(seed_num)

    #pdb.set_trace()
    csv_path = data_csv
    
    total_list = []
    csv_file = os.path.join(csv_path)
    #with open(csv_file, newline='',encoding='cp949') as f:
    with open(csv_file, newline='') as f:
        reader = csv.reader(f)
        tmp = list(reader)
        total_list += tmp
        f.close()

    #pdb.set_trace()
    class_dict = {}

    for data in total_list:
        data = data[0]
        class_name = data.split('/')[-2]
        
        if class_name not in class_dict.keys():
            class_dict[class_name] = []
        class_dict[class_name].append(data)
    
    #pdb.set_trace()

    train_list,val_list = [],[]

    for class_name in class_dict.keys():
        temp_list = class_dict[class_name]
        random.shuffle(temp_list)

        split_idx = int(split_ratio*len(temp_list))
        train_list += temp_list[:split_idx]
        val_list += temp_list[split_idx:len(temp_list)]

    #pdb.set_trace()
    
    return train_list, val_list



def make_encoder(name, input_size=224, input_channels=3, pretrained=True, pretrain_path=None):
    """Make encoder (backbone) with a given name and parameters"""
    
    features_size = input_size // 32
    num_features = 2048
    # if name.startswith('resnet'):
    if name == 'resnet50':
        model = resnet50(pretrained=True)
        features = nn.Sequential(*list(model.children())[:-2])
        # features[0] = nn.Conv2d(in_channels=4, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        num_features = 512 if int(name[6:]) < 50 else 2048
        
        features_shape = (num_features, features_size, features_size)
        return Encoder(model, features, features_shape)

    elif name == 'resnet101':
        model = resnet101(pretrained=True)
        features = nn.Sequential(*list(model.children())[:-2])
        #num_features = 512 if int(name[6:]) < 50 else 2048
        #features_shape = (num_features, features_size, features_size)
        return features#Encoder(model, features, features_shape) 


    elif name == 'resnet18':
        print('resnet18')
        model = resnet18(pretrained=True)
        features = nn.Sequential(*list(model.children())[:-3])
        # features_shape = (num_features, features_size, features_size)
        return features #Encoder(model, features, features_shape) 




    elif name == 'resnet2p1d':
        model = resnet2p1d.generate_model(model_depth=18)
                                        #   n_classes=opt.n_classes,
                                        #   n_input_channels=opt.n_input_channels,
                                        #   shortcut_type='B',
                                        #   conv1_t_size=opt.conv1_t_size,
                                        #   conv1_t_stride=opt.conv1_t_stride,
                                        #   no_max_pool=opt.no_max_pool,
                                        #   widen_factor=opt.resnet_widen_factor)
        
        # model_without_last = nn.Sequential(*list(model.children())[:-1]).state_dict()
        #model_dict = model.state_dict()
        #pretrained_dict = torch.load(pretrain_path, map_location='cpu')['state_dict']
        #pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'fc' not in k}
        #model_dict.update(pretrained_dict)
        #model.load_state_dict(model_dict)
        features = nn.Sequential(*list(model.children())[:-3])
        #features[0] = nn.Conv3d(4, 110, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False)
        return features
        # features = model

    else:
        raise KeyError("Unknown model name: {}".format(name))


        
    # elif name.startswith('mobilenetv2'):
    #     model = mobilenetv2.MobileNetV2(input_size=input_size, pretrained=None)
    #     features = model.features
    #     num_features = 1280
    # elif name.startswith('rmnet'):
    #     model = rmnet.RMNetClassifier(1000, pretrained=None)
    #     features = nn.Sequential(*list(model.children())[:-2])
    #     num_features = 512
        
    # elif name.startswith('se_res'):
    #     model = load_from_pretrainedmodels(name)(pretrained='imagenet' if pretrained else None)
    #     features = nn.Sequential(*list(model.children())[:-2])




def load_from_pretrainedmodels(model_name):
    import pretrainedmodels
    return getattr(pretrainedmodels, model_name)



def squash_dims(tensor, dims):
    """
    Squashes dimension, given in dims into one, which equals to product of given.

    Args:
        tensor (Tensor): input tensor
        dims: dimensions over which tensor should be squashed

    """
    assert len(dims) >= 2, "Expected two or more dims to be squashed"

    size = tensor.size()

    squashed_dim = size[dims[0]]
    for i in range(1, len(dims)):
        assert dims[i] == dims[i - 1] + 1, "Squashed dims should be consecutive"
        squashed_dim *= size[dims[i]]

    result_dims = size[:dims[0]] + (squashed_dim,) + size[dims[-1] + 1:]
    return tensor.contiguous().view(*result_dims)


def unsquash_dim(tensor, dim, res_dim):
    """
    Unsquashes dimension, given in dim into separate dimensions given is res_dim
    Args:
        tensor (Tensor): input tensor
        dim (int): dimension that should be unsquashed
        res_dim (tuple): list of dimensions, that given dim should be unfolded to

    """
    size = tensor.size()
    result_dim = size[:dim] + res_dim + size[dim + 1:]
    return tensor.view(*result_dim)
