# -*- coding: utf-8 -*-
""" モデルを訓練するモジュール """
import time
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
# 学習用データと評価用データを分割
from torch.utils.data.sampler import SubsetRandomSampler
from torchsummary import summary

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# 訓練の進捗バーを表示
import tqdm
# 訓練の様子を可視化
from tensorboardX import SummaryWriter

import variables as v
import dataset
import net
import validation


def train_net(train_loader, eval_loader, writer=None, model=net.VoNet_origin):
    """ ネットワークにデータローダから画像を入力し学習する """
    train_losses = []
    # 訓練データの予測精度
    train_accuracy = []
    # 検証データの予測精度
    val_accuracy = []
    # optimizer = optim.Adam(model.parameters(), lr=0.01)
    optimizer = optim.Adam([
        {'params': model.features.parameters()},
        {'params': model.classifier.parameters(), 'weight_decay': 0.1}
    ], lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(v.EPOCHS):

        # 訓練モードにする
        model.train()

        running_loss = 0.0
        n = 0
        n_accuracy = 0

        for i, data in tqdm.tqdm(enumerate(train_loader, 0), total=len(train_loader)):
            inputs, labels = data[0].to(v.DEVICE), data[1].to(v.DEVICE)
            # 勾配を初期化
            optimizer.zero_grad()
            # ネットワークにデータを入力
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # 誤差逆伝播で勾配を更新
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n += len(inputs)

            # 確率が最大のクラスを予測
            _, y_pred = outputs.max(1)
            n_accuracy += (labels == y_pred).float().sum().item()

        train_losses.append(running_loss / i)
        train_accuracy.append(n_accuracy / n)
        val_accuracy.append(validation.validation_net(eval_loader, model))
        print('[%2d] loss: %.4f acc(train): %.4f acc(val): %.4f' %
              (epoch + 1, train_losses[-1], train_accuracy[-1], val_accuracy[-1]),
              flush=True)
        # print('[%2d] loss: %.4f' % (epoch + 1, train_losses[-1]), flush=True)

        if writer:
            writer.add_scalar("train_loss", train_losses[-1], epoch)
            writer.add_scalars("accuracy", {
                "train": train_accuracy[-1],
                "validation": val_accuracy[-1]
            }, epoch)

    _train_losses_str = [str(n) for n in train_losses]
    _train_acc_str    = [str(n) for n in train_accuracy]
    _val_acc_str      = [str(n) for n in val_accuracy]

    with open(v.TRAIN_LOSS_PATH, mode='w') as f_stream:
        f_stream.write('\n'.join(_train_losses_str))
        print('train_losses written in {}'.format(v.TRAIN_LOSS_PATH))
    with open(v.TRAIN_ACC_PATH, mode='w') as f_stream:
        f_stream.write('\n'.join(_train_acc_str))
        print('train_accuracy written in {}'.format(v.TRAIN_ACC_PATH))
    with open(v.VAL_ACC_PATH, mode='w') as f_stream:
        f_stream.write('\n'.join(_val_acc_str))
        print('bal_accuracy written in {}'.format(v.VAL_ACC_PATH))


def train(model_path:str, use_net=net.VoNet_origin):
    """ データローダの作成からモデルの学習、保存までを行う """
    train_dataset = dataset.MyCustomDataset(
        v.TRAIN_IMG_DIR,
        v.TRAIN_CSV,
        dataset.data_transforms['train'],
    )

    # 学習用データと評価用データを分割
    _dataset_size = len(train_dataset)
    _indices = list(range(_dataset_size))
    _split = int(np.floor(v.VAL_SPLIT * _dataset_size))

    if v.SHUFFLE_DATASET:
        np.random.seed(v.RAND_SEED)
        np.random.shuffle(_indices)

    _train_indices = _indices[_split:]
    _eval_indices = _indices[:_split]

    _train_sampler = SubsetRandomSampler(_train_indices)
    _eval_sampler = SubsetRandomSampler(_eval_indices)

    # ミニバッチに分割する
    train_loader = DataLoader(train_dataset, batch_size=v.BATCH_SIZE, sampler=_train_sampler)
    eval_loader = DataLoader(train_dataset, batch_size=v.BATCH_SIZE, sampler=_eval_sampler)

    # モデルの指定してCPU or GPUに転送
    model = use_net
    model.to(v.DEVICE)

    # ネットワーク構造の書き出し (torchsummary)
    summary(model, (3, v.IMG_WIDTH, v.IMG_WIDTH))

    # ログの格納場所 (tensorboard)
    writer = SummaryWriter(v.LOG_DIR)

    # 学習
    _start = time.time()
    print('Training started.')
    print('# of images: {}\nwidth of images: {}\n# of epochs: {}\nBatch size: {}'
                .format(v.IMG_NUM, v.IMG_WIDTH, v.EPOCHS, v.BATCH_SIZE))
    print('----------------------------------------------------------------')
    train_net(train_loader, eval_loader, writer=writer, model=model)
    print('Elapsed time(training): {} ms.'.format((time.time() - _start) * 1000))
    print('----------------------------------------------------------------')

    # 評価
    # _start = time.time()
    # validation_net(model, eval_loader)
    # print('Elapsed time(evaluation): {} ms.'.format((time.time() - _start) * 1000))
    # print('----------------------------------------------------------------')

    # 学習モデルの保存
    torch.save(model.state_dict(), model_path)
    print('Model saved on {}.'.format(model_path))
