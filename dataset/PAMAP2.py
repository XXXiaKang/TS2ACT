import os
import sys
sys.path.append("..")
import torch
from PIL import Image
import numpy as np
import pandas as pd
import random
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
import numpy as np
import pandas as pd
from clip.clip import *
from tsaug import TimeWarp, Crop, Quantize, Drift, Reverse

class Get_dataset_aug(Dataset):
    def __init__(self,
                 X,
                 y,
                 split = 'train',
                 clip_len = 500):
        super(Dataset, self).__init__()
        self.clip_len = clip_len
        self.split = split
        self.len = X.shape[1]
        self.train_augmenter = (
            TimeWarp()
            # + Crop(size=clip_len) 
            + Quantize(n_levels=[10, 20, 30])
            + Drift(max_drift=(0.1, 0.5)) @ 0.8
            + Reverse() @ 0.5
        )
        self.test_augmenter = (
            Crop(size=clip_len)
        )
        self.X = X.astype(np.float64)
        self.y = y.astype(np.int32)
    def __getitem__(self, index):
        x = self.X[index][np.newaxis,:,:]
        y = self.y[index]
        start1 = random.randint(0, self.len-self.clip_len) if self.len > self.clip_len else 0
        start2 = random.randint(0, self.len-self.clip_len) if self.len > self.clip_len else 0
        if self.split == 'TS2ACT':
            return self.train_augmenter.augment(x)[0,start1:start1+self.clip_len],\
                self.train_augmenter.augment(x)[0,start2:start2+self.clip_len],y
        else:
            return x[0,start2:start2+self.clip_len],y
    def __len__(self):
        return len(self.y)



class Get_dataset(Dataset):
    def __init__(self,
                 X,
                 y,
                 split = 'train',
                 clip_len = 500):
        super(Dataset, self).__init__()
        self.clip_len = clip_len
        self.split = split
        self.len = X.shape[1]
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y).view(-1).long()
    def __getitem__(self, index):
        if self.split == 'TS2ACT':
            start1 = random.randint(0, self.len-self.clip_len) if self.len > self.clip_len else 0
            start2 = random.randint(0, self.len-self.clip_len) if self.len > self.clip_len else 0
            return  self.X[index][start1:start1+self.clip_len], \
                    self.X[index][start2:start2+self.clip_len], \
                    self.y[index]
        else:
            start = random.randint(0, self.len-self.clip_len) if self.len > self.clip_len else 0
            return self.X[index][start:start+self.clip_len], self.y[index]
    def __len__(self):
        return len(self.y)

def PAMAP_TS2ACT_aug(clip_len = 500 ,dataset_dir='/home/xiakang/data/PAMAP2_Dataset',name="1-shot"):
    path = os.path.join(dataset_dir, name)
    xtrain = np.load(os.path.join(path,"xtrain.npy"))
    xtest = np.load(os.path.join(path,"xtest.npy"))
    ytrain = np.load(os.path.join(path,"ytrain.npy"))
    ytest = np.load(os.path.join(path,"ytest.npy"))
    print('\n---------------------------------------------------------------------------------------------------------------------\n')
    print('xtrain shape: %s\nxtest shape: %s\nytrain shape: %s\nytest shape: %s'%(xtrain.shape, xtest.shape, ytrain.shape, ytest.shape))
    return Get_dataset( xtrain, ytrain, clip_len = clip_len), Get_dataset(xtest, ytest,'TS2ACT',clip_len = clip_len)



def test():
    # os.chdir('/home/xiakang/xiak/TS')
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    prompt = 'an action of'
    labels = ['laying','sitting','standing','walking','running','cycling','Nordic walking','ascending stairs','descending stairs','vacuum cleaning','ironing','rope jumping',]
    prompt_labels = [  prompt + 'laying',
                prompt + 'sitting',
                prompt + 'standing',
                prompt + 'walking',
                prompt + 'running',
                prompt + 'cycling',
                prompt + 'Nordic walking',
                prompt + 'ascending stairs',
                prompt + 'descending stairs',
                prompt + 'cleaning vacuum',
                prompt + 'ironing',
                prompt + 'jumping rope',
            ]
    text = tokenize(prompt_labels).to(device)
    model, preprocess = load("../output/ViT-B-32.pt")
    model = model.to(device)
    model.eval()
    with torch.no_grad(): text_features = model.encode_text(text)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    torch.save(text_features,"/home/xiakang/data/PAMAP2_Dataset/PAMAP_text.pth")
    save = []
    for label in labels:
        images = []
        for i in range(50):
            images.append(preprocess(Image.open('/home/xiakang/data/PAMAP2_Dataset/'+label+'/' + 
                str(i+1)+".jpg")))
        images = torch.tensor(np.stack(images)).to(device)
    
        with torch.no_grad():
            image_features = model.encode_image(images)
            save.append(image_features)
    image_features = torch.stack(save,dim=0)
    torch.save(image_features,"/home/xiakang/data/PAMAP2_Dataset/PAMAP_image.pth")
    print(image_features.shape)


if __name__ == '__main__':
    test()
