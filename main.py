# -*- coding: utf-8 -*-
""" main
"""
import os
from torchvision.models import SqueezeNet

import dataset
import net
import train
import validation
import variables as v
import grapher


def make_dir(dirname:str=v.DATE_STR):
    _path = 'model/{}'.format(dirname)
    if not os.path.exists(_path):
        os.makedirs(_path)
        print('directory {} created.'.format(_path))


if __name__ == '__main__':
    use_net = net.VoNet_origin()
    make_dir()

    #dataset.create_dataset_compcars(v.TRAIN_TXT, v.TRAIN_CSV)
    #dataset.create_dataset_compcars(v.TEST_TXT, v.TEST_CSV)
    
    dataset.create_dataset_custom()
    dataset.create_dataset_custom()

    "確認用画像の生成と出力"
    #dataset.showimage(v.EXAMPLE_OUTPUT_PATH)
    #dataset.saveimage()
    train.train(v.MODEL_PATH, use_net)
    validation.validation_dataloader(v.MODEL_PATH, use_net)
    validation.validation_custom('dir', 'evaluation/original/', 'labels', use_net)
    validation.validation_custom('csv', 'evaluation/original/', 'labels', use_net)
    validation.validation_custom_dataset(use_net=use_net)
    grapher.grapher(grapher.losslist(v.DATE_STR), grapher.acclist(v.DATE_STR), v.DATE_STR)