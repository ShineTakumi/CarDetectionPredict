# -*- coding: utf-8 -*-
""" dataset.py
"""
import random

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import variables as v

# データオーグメンテーションの設定
data_transforms = {
    'train': transforms.Compose([
        #transforms.RandomHorizontalFlip(),
        # transforms.RandomAffine(degrees=[-10.0, 10.0], translate=(0.2, 0.2), scale=(1.2, 1.5), fillcolor=128),
        # transforms.RandomPerspective(distortion_scale=0.3, p=0.5, interpolation=3, fill=128),
        transforms.RandomResizedCrop(v.IMG_WIDTH / random.uniform(1.0, 4.0), (1.0, 1.0)),
        transforms.Resize((v.IMG_WIDTH, v.IMG_WIDTH)), # 入力画像サイズを統一するため必須
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.4887, 0.4765, 0.4674], std=[0.2848, 0.2835, 0.2888]),
        # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.0),
        transforms.ColorJitter(brightness=0.01, contrast=0.01, saturation=0.01, hue=0.0),
        # transforms.Grayscale(), # グレースケールで学習する場合
    ]),
    'val': transforms.Compose([
        transforms.Resize((v.IMG_WIDTH, v.IMG_WIDTH)), # 入力画像サイズを統一するため必須
        transforms.ToTensor(),
        # transforms.Grayscale(), # グレースケールで評価する場合
    ]),
    'train-test': transforms.Compose([
        transforms.RandomResizedCrop(v.IMG_WIDTH / random.uniform(1.0, 4.0), (1.0, 1.0)),
        transforms.Resize((v.IMG_WIDTH, v.IMG_WIDTH)),
    ])
}


def create_dataset_compcars(txt_path, csv_path):
    """ CompCarsに用意されている学習・識別用データセットを利用する場合 """
    _read_data = []
    _label_vals = []
    _line_count = 0
    img_num = v.IMG_NUM

    print('Creating csv...')
    with open(txt_path, mode='rt') as f_stream:
        for line in f_stream:
            if _line_count < img_num:
                _read_data.append(line.split('\n', 1)[0].split('.')[0])
                _line_count += 1
            else:
                break
    for i in _read_data:
        #with open('{}/data/label/{}.txt'.format(v.DATASET_DIR, i), mode='rt') as f_stream:
        with open('{}/data/label/{}.txt'.format(v.DATASET_DIR, i), mode='rt') as f_stream:
            _label_vals.append(f_stream.readlines()[0].split("\n", 1)[0])
    _label_dict = dict(zip(_read_data, _label_vals))
    data_frame = pd.DataFrame(list(_label_dict.items()), columns=["path", "label"])
    data_frame.to_csv(csv_path)
    print('----------------------------------------------------------------')
    print("csv saved on {}. \n{}\n".format(csv_path, data_frame))


def create_dataset_custom():
    """ 学習・識別用データセットを自作する場合 """
    _keys, _vals = [], []
    _val0, _val1, _val2, _val3, _val4, _val5, _val6, _val7, _val8 = 0, 0, 0, 0, 0, 0, 0, 0, 0

    print('Creating csv...')
    # データセットを作成
    for ort in v.LBL_TXT[:v.IMG_NUM]:
        with open(ort) as f_stream:
            label_val = f_stream.readlines()[0].split("\n", 1)[0]
            # ラベルの値が"-1"の画像は向き不明なので除外
            if label_val != "-1":
                #_keys.append(ort.split('.', 1)[1].split("{}/".format(v.TRAIN_LBL_DIR), 1)[1])
                _keys.append(ort.split("{}/".format(v.TRAIN_LBL_DIR), 1)[1].split(".",1)[0])

                _vals.append(label_val)

                if label_val == '1':
                    _val1 += 1
                elif label_val == '2':
                    _val2 += 1
                elif label_val == '3':
                    _val3 += 1
                elif label_val == '4':
                    _val4 += 1
                elif label_val == '5':
                    _val5 += 1
                elif label_val == '6':
                    _val6 += 1
                elif label_val == '7':
                    _val7 += 1
                elif label_val == '8':
                    _val8 += 1
            else:
                _val0 += 1

    # 各ラベルの総数を出力
    print('label==1:  {}'.format(_val1))
    print('label==2:  {}'.format(_val2))
    print('label==3:  {}'.format(_val3))
    print('label==4:  {}'.format(_val4))
    print('label==5:  {}'.format(_val5))
    print('label==6:  {}'.format(_val6))
    print('label==7:  {}'.format(_val7))
    print('label==8:  {}'.format(_val8))
    print('label==-1: {}'.format(_val0))
    # 画像Pathとラベルの辞書をCSVで書き出し
    label_dict = dict(zip(_keys, _vals))
    data_frame = pd.DataFrame(list(label_dict.items()), columns=["path", "label"])
    data_frame.to_csv(v.TRAIN_CSV)
    print('----------------------------------------------------------------')
    print("csv saved on {}. \n{}\n".format(v.TRAIN_CSV, data_frame))


class MyCustomDataset(Dataset):
    """ CSVファイルから画像を読み込み変換をかける """
    def __init__(self, image_dir, csv_path, transform=None):
        self.data_frame = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, index):
        # CSVから画像Pathとラベルを読み込む
        path = self.data_frame["path"][index]
        label = int(self.data_frame["label"][index])
        image = Image.open("{}/{}.jpg".format(self.image_dir, path))
        if self.transform:
            image = self.transform(image)

        return (image, label)


def get_mean_std():
    """
    DataLoader からデータセット全体の平均と標準偏差を求める
    Returns:
        (mean->torch.Tensor, std->torch.Tensor)
    Tips:
        CompCars16,016枚の学習用データの平均と標準偏差:
        mean: tensor([0.4887, 0.4765, 0.4674])
        std: tensor([0.2848, 0.2835, 0.2888])
    """

    train_dataset = MyCustomDataset(
        v.TRAIN_IMG_DIR,
        v.TRAIN_CSV,
        transform=transforms.Compose([
            transforms.RandomResizedCrop(size=512, scale=(1.0, 1.0)),
            transforms.ToTensor()
        ])
    )

    loader = DataLoader(dataset=train_dataset, batch_size=v.BATCH_SIZE, shuffle=False)
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0

    for data, _ in loader:
        channels_sum += torch.mean(data, dim=[0,2,3])
        channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean**2) ** 0.5

    print('mean: {}\nstd: {}'.format(mean, std))
    return mean, std


def showimage(filename: str):
    """ 4つ並べた画像を生成 データオーグメンテーション比較用 """
    img = Image.open('model/transform-example.jpg')
    plt.subplot(1, 4, 1)
    plt.imshow(np.array(img))

    _transform = transforms.Compose([
        transforms.ToPILImage()
    ])

    trans_imgs = []
    for i in range(3):
        img_transformed = data_transforms['train'](img)
        img_PILed = _transform(img_transformed)
        trans_imgs.append(img_PILed)
        plt.subplot(1, 4, i+2)
        plt.imshow(np.array(trans_imgs[i]), cmap = 'gray')

    for i in range(4):
        plt.subplot(1, 4, i+1)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


def saveimage():
    """ data_transformsのデータ拡張による変換後の画像を保存する """
    img = Image.open('model/transform-example.jpg')
    img = data_transforms['train-test'](img)
    img.save('saveimage.png')