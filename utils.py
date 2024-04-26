import os
import torch
import argparse
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader
from models.net import *
device = torch.device('cuda')
def load_text(opt):
    if opt.dataset == 'PAMAP2':
        clip_text = torch.load(os.path.join('/home/xiakang/data/PAMAP2_Dataset', "PAMAP_text.pth")).float()
    elif opt.dataset == 'UCI':
        clip_text = torch.load(os.path.join('/home/xiakang/data/UCI-HAR/UCI HAR Dataset', "UCI_text.pth")).float()
    elif opt.dataset == 'MotionSense':
        clip_text = torch.load(os.path.join('/home/xiakang/data/MotionSense', "MotionSense_text.pth")).float()
    elif opt.dataset == 'HHAR':
        clip_text = torch.load(os.path.join('/home/xiakang/data/HHAR', "HHAR_text.pth")).float()
    else:
        return None
    clip_text = (clip_text / clip_text.norm(dim=-1, keepdim=True)).to(device)
    return clip_text

def load_image(opt):
    if opt.dataset == 'PAMAP2':
        clip_image = torch.load(os.path.join('/home/xiakang/data/PAMAP2_Dataset', "PAMAP_image.pth")).float().to(device)
    elif opt.dataset == 'UCI':
        clip_image = torch.load(os.path.join('/home/xiakang/data/UCI-HAR/UCI HAR Dataset', "UCI_image.pth")).float().to(device)
    elif opt.dataset == 'MotionSense':
        clip_image = torch.load(os.path.join('/home/xiakang/data/MotionSense', "MotionSense_image.pth")).float().to(device)
    elif opt.dataset == 'HHAR':
        clip_image = torch.load(os.path.join('/home/xiakang/data/HHAR', "HHAR_image.pth")).float().to(device)
    else:
        return None
    clip_image = clip_image / clip_image.norm(dim=-1, keepdim=True)
    return clip_image

