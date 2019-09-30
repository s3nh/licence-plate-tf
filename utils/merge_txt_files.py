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
    with open(outfile, 'w') as o:
        for _file in _files:
            with open(_file, 'r') as f:
                _tmp = [line.split('\t') for line in f]
                _tmp = [el for sublist in _tmp for el in sublist]
                o.write('{}\n'.format(','.join([el for el in _tmp] )))
    o.close()

def main():
    PATH = 'dataset/benchmarks/endtoend/eu'
    get_files(PATH, '*.txt', outfile='outpath.csv')
    _file = pd.read_csv('outpath.csv')
    _file.columns = ['filename', 'x_coord', 'y_coord', 'height', 'width', 'licence' ]
    _file.to_csv('labels.csv', index=False)  

if __name__ == "__main__":
    main()    
    