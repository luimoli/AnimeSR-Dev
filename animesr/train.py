# flake8: noqa
import os,sys
sys.path.append(os.getcwd())



import os.path as osp
import animesr.archs
import animesr.data
import animesr.models
from basicsr.train import train_pipeline

import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)
