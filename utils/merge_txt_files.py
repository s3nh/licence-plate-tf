import numpy as np 
import os 
import re 
import pandas as pd 
import json 
import argparse
import glob

PATH = 'dataset/benchmarks/endtoend/eu'


def build_args():
    
    parser= argparse.ArgumentParser()
    parser.add_argument('input_path', type = str, help = 'Input path')
    parser.add_argument('')


def get_files(_path, extension, outfile):
    out_list = []
    os.chdir(_path)
    _files = glob.glob(extension)
    for _file in _files:
        with open(_file, 'r') as f:
            _tmp = [line.split('\t') for line in f]
            _tmp.append(out_list)
    return out_list




    
    
