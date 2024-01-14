'''
@Author: M.H.C.
@Desc: Tensor operations.
'''
import torch
from typing import Any
from torch import Tensor


def cat_tensors(tensor_a:Any, tensor_b:Tensor) -> Tensor:
    if tensor_a is None:
        return tensor_b
    else:
        assert len(tensor_a.shape) == len(tensor_b.shape), 'The shape of Tensor A and B must be the same!'
        return torch.cat((tensor_a, tensor_b), 0)