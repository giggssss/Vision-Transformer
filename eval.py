from __future__ import print_function
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

import glob
import os
import torch
import shutil
import random
import requests
from io import BytesIO
import argparse

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


def parse_args():
    parser = argparse.ArgumentParser(description='Day and Night Classifier')
    parser.add_argument('--data_type', default='u', help='data type f(file), u(url) or d(dir)')
    parser.add_argument('--data_path', default=r'D:\Data\train', help='image file path dir or url')
    parser.add_argument('--checkpoint', default='save2_gpu.pt', help='checkpoint file')
    parser.add_argument('--print', default=False, action='store_true', help='print on/off')
    parser.add_argument('--file_copy', default=False, action='store_true', help='file_copy on/off')

    args = parser.parse_args()
    return args


def file_list_load(path):
    data_list = glob.glob(os.path.join(path, '*.jpg'))
    # evl_list = glob.glob(os.path.join(path, "/**/*.jpg"), recursive=True)
    random.shuffle(data_list)

    return data_list


def file_load(path, data_type):
    totensor = transforms.ToTensor()
    if data_type == 'u':
        response = requests.get(path)
        img = Image.open(BytesIO(response.content))
        img = img.resize((256, 256))
        image = totensor(img).unsqueeze(0).to(device)
    elif data_type == 'f' or data_type == 'd':
        img = Image.open(path, 'r')
        img = img.resize((256, 256))
        image = totensor(img).unsqueeze(0).to(device)
    else:
        print('file type error!!!')

    return image


if __name__ == '__main__':
    args = parse_args()
    model = torch.load(args.checkpoint, map_location=torch.device(device))
    model.eval()

    if args.data_type == 'f' or args.data_type == 'u':
        input_im = file_load(args.data_path, args.data_type)
        output = model(input_im)
        pred = int(output.argmax(dim=1))
        conf = float(output.max(dim=1).values)
        print(f"{os.path.basename(args.data_path)} --> {'Day' if pred == 0 else 'Night'} || Confidence : {conf}")

    elif args.data_type == 'd':
        data_list = file_list_load(args.data_path)
        for file in tqdm(data_list):
            input_im = file_load(file, args.data_type)
            output = model(input_im)
            pred = int(output.argmax(dim=1))
            conf = float(output.max(dim=1).values)
            diff = float(output.max(dim=1).values) - float(output.min(dim=1).values)

            acc = 0
            label = -1

            if len(os.path.basename(file).split('_')) > 1:
                label = int(os.path.basename(file).split('_')[0])
                if pred == label:
                    acc += 1.0
                else:
                    if args.file_copy:
                        shutil.copy(file, r'D:\Data\res_error' + f'\\{pred}_{conf:.6f}.jpg')

                if args.print:
                    print(label, pred, conf, diff, os.path.basename(file))

            else:
                if args.print:
                    print(f"{os.path.basename(file)} --> {'Day' if pred == 0 else 'Night'} || Confidence : {conf}")
            if args.file_copy:
                if conf < 1.0:
                    shutil.copy(file, r'D:\Data\res_rowconf'+f'\\{pred}_{conf:.6f}.jpg')
                elif pred == 0:
                    shutil.copy(file, r'D:\Data\res_day'+f'\\{pred}_{conf:.6f}.jpg')
                elif pred == 1:
                    shutil.copy(file, r'D:\Data\res_night'+f'\\{pred}_{conf:.6f}.jpg')

        accuracy = acc / len(data_list)
        print(f'- acc: {accuracy:.4f} = {acc}/{len(data_list)}\n')

    else:
        print('file type error!!!')
