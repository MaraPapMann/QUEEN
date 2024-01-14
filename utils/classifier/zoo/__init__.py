import pathlib
import sys
import os
# sys.path.append('.')
from .LeNet import *
from .densenet import *
from .resnet import *
from .vgg import *
from .inception import *
from .mobilenetv2 import *
from .googlenet import *
from .ConvNet import conv_net
from .AlexNet import alex_net
from .FullyConnected import FullyConnected


def get_cls(model_arch:str, num_classes:int, device:str='cuda:0', **kwargs):
    if model_arch == 'LeNet':
        cls = lenet(num_classes=num_classes, **kwargs)
    elif model_arch == 'FullyConnected':
        cls = FullyConnected(num_classes=num_classes, **kwargs)
    elif model_arch == 'ResNet18':
        cls = resnet18(num_classes=num_classes, **kwargs)
    elif model_arch == 'ResNet34':
        cls = resnet34(num_classes=num_classes, **kwargs)
    elif model_arch == 'ResNet50':
        cls = resnet50(num_classes=num_classes, **kwargs)
    elif model_arch == 'DenseNet121':
        cls = densenet121(num_classes=num_classes, **kwargs)
    elif model_arch == 'DenseNet161':
        cls = densenet161(num_classes=num_classes, **kwargs)
    elif model_arch == 'DenseNet169':
        cls = densenet169(num_classes=num_classes, **kwargs)
    elif model_arch == 'VGG11-BN':
        cls = vgg11_bn(num_classes=num_classes, **kwargs)
    elif model_arch == 'VGG13-BN':
        cls = vgg13_bn(num_classes=num_classes, **kwargs)
    elif model_arch == 'VGG16-BN':
        cls = vgg16_bn(num_classes=num_classes, **kwargs)
    elif model_arch == 'VGG19-BN':
        cls = vgg19_bn(num_classes=num_classes, **kwargs)
    elif model_arch == 'InceptionNet':
        cls = inception_v3(num_classes=num_classes, **kwargs)
    elif model_arch == 'MobileNet':
        cls = mobilenet_v2(num_classes=num_classes, **kwargs)
    elif model_arch == 'GoogleNet':
        cls = googlenet(num_classes=num_classes, **kwargs)
    elif model_arch == 'ConvNet':
        cls = conv_net(num_classes=num_classes, **kwargs)
    elif model_arch == 'AlexNet':
        cls = alex_net(num_classes=num_classes, **kwargs)
    else:
        raise Exception('Enter a valid model architecture!')
    
    cls = cls.to(device)
    return cls


'''Debug'''
if __name__ == '__main__':
    lst_model_arch = ['LeNet', 'ResNet18', 'ResNet34', 'ResNet50', 'DenseNet121', 'DenseNet161', 'DenseNet169', 'VGG11-BN', 'VGG13-BN', 'VGG16-BN', 'VGG19-BN', 'InceptionNet', 'MobileNet', 'GoogleNet']
    num_classes = 10
    channel = 3
    im_size = (32, 32)
    depth = 3
    res = 32
    
    for model_arch in lst_model_arch:
        if model_arch == 'ConvNet':
            cls = get_cls(model_arch, num_classes, depth=depth, channel=channel, im_size=im_size)
        elif model_arch == 'AlexNet':
            cls = get_cls(model_arch, num_classes, res)
        cls = get_cls(model_arch, num_classes)
        print(cls)