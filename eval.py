from __future__ import print_function

import glob
import os
import torch
import shutil
import torch.nn as nn
import torch.optim as optim
from linformer import Linformer
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm


from vit_pytorch.efficient import ViT

device = 'cpu'
os.makedirs('data', exist_ok=True)

train_dir = 'data/train'
train_list = glob.glob(os.path.join(train_dir, '*.jpg'))

model = torch.load('save.pt')
model.eval()

totensor = transforms.ToTensor()
for file in train_list:
    img = Image.open(file, 'r')
    img = img.resize((256, 256))
    input = totensor(img).unsqueeze(0)
    output = model(input)
    pred = int(output.argmax(dim=1))
    conf = float(output.max(dim=1).values)
    diff = float(output.max(dim=1).values) - float(output.min(dim=1).values)
    print(os.path.basename(file).split('_')[0], pred, conf, diff, os.path.basename(file))

    # if pred != int(os.path.basename(file).split('_')[0]):
    #     shutil.copy(file, 'data/res_error')
    #     print(os.path.basename(file).split('_')[0], pred, conf, diff, os.path.basename(file))

