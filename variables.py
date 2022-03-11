# -*- coding: utf-8 -*-
""" 変数、定数を定義 """
import glob
import torch

DATASET_DIR   = '../datasets/CompCars'
TRAIN_IMG_DIR = DATASET_DIR + '/data/image'
TRAIN_LBL_DIR = DATASET_DIR + '/data/label'
LBL_TXT       = sorted(glob.glob('{}/**/**/**/*.txt'.format(TRAIN_LBL_DIR)))

#学習用又はテスト用の画像のｐディレクトリを羅列したtxtファイル
TRAIN_TXT     = DATASET_DIR + '/data/train_test_split/classification/train.txt'
TEST_TXT      = DATASET_DIR + '/data/train_test_split/classification/test.txt'

TRAIN_CSV     = 'csv/train.csv'
TEST_CSV      = 'csv/test.csv'
LOG_DIR       = 'tensorboard'     # TensorBoardXの書き込み先

def available_cuda():
    """ GPUが利用可能ならばGPU上で動作 """
    return 'cuda:0' if torch.cuda.is_available() else 'cpu'
DEVICE = torch.device(available_cuda())

IMG_NUM    = 1600       # 学習データセットの枚数
#IMG_NUM    = 1920 
IMG_WIDTH  = 113        # 学習データセットの画像サイズ
#EPOCHS     = 30
EPOCHS     = 60       # エポック数 (データセット全体を何回学習させるか)
BATCH_SIZE = 64         # バッチサイズ (データセットを何枚ごとのミニバッチに分割するか)
#BATCH_SIZE = 256
VAL_SPLIT = 0.25        # 学習データセットの画像割合
SHUFFLE_DATASET = False
#SHUFFLE_DATASET = True  # DataLoader作成時データセットをごちゃまぜにするか
RAND_SEED = 42

""" train.pyの設定 """
DATE_STR            = 'test-8-1215'
MODEL_PATH          = 'model/{0}/model-{0}.pt'.format(DATE_STR)
EXAMPLE_OUTPUT_PATH = 'model/{0}/example-{0}.png'.format(DATE_STR)
TRAIN_LOSS_PATH     = 'model/{0}/loss-{0}.txt'.format(DATE_STR)
TRAIN_ACC_PATH      = 'model/{0}/acc(train)-{0}.txt'.format(DATE_STR)
VAL_ACC_PATH        = 'model/{0}/acc(val)-{0}.txt'.format(DATE_STR)

""" validation.py の設定 """
CUSTOM_VAL_PATHS    = [
    'evaluation/original',
    'evaluation/original-flip',
    'evaluation/sunset',
    'evaluation/sunset-flip',
    'evaluation/night',
    'evaluation/night-flip',
    'evaluation/blur',
    'evaluation/blur-flip',
]
CUSTOM_VAL_OUTPUT_PATH = 'model/{0}/result(custom)-{0}'.format(DATE_STR)
