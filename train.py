from __future__ import print_function

import glob
import os
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from linformer import Linformer
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm

# from vit_pytorch.efficient import ViT
from vit_pytorch.vit_for_small_dataset import ViT

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


class Dataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)
        img_transformed = self.transform(img)

        anno = int(os.path.basename(img_path).split('_')[0])

        return img_transformed.to(device), anno


def parse_args():
    parser = argparse.ArgumentParser(description='Day and Night Classifier')
    parser.add_argument('--path', default=r'D:\Data\train', help='image file path dir or url')
    parser.add_argument('--batchs', default=64, help='image file path dir or url')
    parser.add_argument('--epochs', default=10, help='image file path dir or url')
    parser.add_argument('--lr', default=3e-5, help='image file path dir or url')
    parser.add_argument('--gamma', default=0.7, help='image file path dir or url')
    parser.add_argument('--seed', default=42, help='image file path dir or url')
    parser.add_argument('--checkpoint', default='save2_gpu.pt', help='checkpoint file')

    args = parser.parse_args()
    return args


def get_dataloader(args):
    data_list = file_list_load(args.path)
    labels = [os.path.basename(args.path).split('_')[0] for path in data_list]
    train_list, valid_list = train_test_split(data_list,
                                              test_size=0.2,
                                              stratify=labels,
                                              random_state=args.seed)

    print(f"Train Data: {len(train_list)}")
    print(f"Validation Data: {len(valid_list)}")

    [train_transforms, test_transforms] = get_transform()

    train_data = Dataset(train_list, transform=train_transforms)
    valid_data = Dataset(valid_list, transform=test_transforms)

    t_loader = DataLoader(dataset=train_data, batch_size=args.batchs, shuffle=True)
    v_loader = DataLoader(dataset=valid_data, batch_size=args.batchs, shuffle=True)

    return [t_loader, v_loader]


def file_list_load(path):
    data_list = glob.glob(os.path.join(path, '*.jpg'))
    return data_list


def get_transform():
    train_transforms = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomChoice([
                transforms.RandomResizedCrop(256),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(30),
                transforms.RandomPerspective()]),
            transforms.ToTensor(),
            transforms.RandomErasing(),
        ]
    )

    val_transforms = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ]
    )
    return [train_transforms, val_transforms]


def get_models(args):
    if args.checkpoint:
        model = torch.load(args.checkpoint, map_location=torch.device(device))
        print(args.checkpoint + ' is loaded....')

    else:
        efficient_transformer = Linformer(
            dim=128,
            seq_len=49+1,  # 7x7 patches + 1 cls-token
            depth=12,
            heads=8,
            k=64
        )
        model = ViT(
            dim=128,
            image_size=256,
            patch_size=16,
            num_classes=2,
            # transformer=efficient_transformer,
            channels=3,
            depth=6,
            heads=16,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1
        ).to(device)
    return model


def get_loss():
    # loss function
    crit = nn.CrossEntropyLoss()
    return crit


def get_scheduler(model, args):
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # scheduler
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    return [optimizer, scheduler]


if __name__ == '__main__':
    args = parse_args()

    best_loss = 1000
    best_acc = 0

    [train_loader, valid_loader] = get_dataloader(args)
    model = get_models(args)
    criterion = get_loss()
    [optimizer, scheduler] = get_scheduler(model, args)

    for epoch in range(args.epochs):
        epoch_loss = epoch_accuracy = 0
        for data, label in tqdm(train_loader):
            output = model(data)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = (output.argmax(dim=1) == label).float().mean()
            epoch_accuracy += acc / len(train_loader)
            epoch_loss += loss / len(train_loader)

        with torch.no_grad():
            epoch_val_accuracy = epoch_val_loss = 0
            for data, label in valid_loader:
                val_output = model(data)
                val_loss = criterion(val_output, label)

                acc = (val_output.argmax(dim=1) == label).float().mean()
                epoch_val_accuracy += acc / len(valid_loader)
                epoch_val_loss += val_loss / len(valid_loader)

        print(
            f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
        )
        if epoch_val_loss < best_loss and epoch_val_accuracy > best_acc:
            torch.save(model, 'save2-1.pt')
            best_loss = epoch_val_loss
            best_acc = epoch_val_accuracy
