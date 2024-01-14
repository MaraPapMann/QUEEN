'''
@Author: M.H.C.
@Desc: An argument processor that saves and loads the argument in .json format.
'''
from typing import *
import argparse


class Args(object):
    '''
    @Desc: An empty class that stores the arguments.
    '''
    def __init__(self) -> None:
        return

    def print_args(self) -> None:
        print(self.__dict__)
        return
    
    def get_args(self) -> dict:
        return self.__dict__


class ArgProcessor(object):
    def __init__(self) -> None:
        pass

    @staticmethod
    def parse_args(arg_parser:argparse.ArgumentParser) -> Args:
        args = dict(vars(arg_parser.parse_args()))  # Namespace -> Dict
        arg_class = Args()
        for k in args.keys():
            arg_class.__setattr__(k, args[k])
        return arg_class
    
    @staticmethod
    def save_json(args:Args) -> None:
        #TODO
        return
    
    @staticmethod
    def load_json(pth_json:str) -> Args:
        #TODO
        return
