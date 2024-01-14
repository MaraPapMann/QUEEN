import os
import torch
from torch import Tensor


def get_files(dir:str, ext:str) -> list:
    """
    @Desc: To get all files in directory with a extension.
    @Params:
        dir: string, path to the directory;
        ext: string, the extension name;
    @Return:
        files: list of strings, all files in the directory with the extension name.
    """
    files = []
    if dir[-1] != '/':
        dir = dir + '/'
    files += [dir + each for each in os.listdir(dir) if each.endswith(ext)]
    return files


def cat_tensor(base:Tensor, to_cat:Tensor) -> Tensor:
    if base == None:
        base = to_cat
    else:
        base = torch.cat((base, to_cat), 0)
        
    return base