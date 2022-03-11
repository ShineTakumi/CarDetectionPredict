# -*- coding: utf-8 -*-
""" 検証用 """
import glob
import time

import pandas as pd
import torch
from torch.utils.data import DataLoader
from PIL import Image

import dataset
import net
import variables as v

def validation_net(data_loader, use_net=net.VoNet_origin):
    """ モデルに画像を入力し車両の向きを推定する """
    use_net.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data in data_loader:
            images, labels = data[0].to(v.DEVICE), data[1].to(v.DEVICE)

            outputs = use_net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = correct / total
    print('acc(val): %2.2f %%' % (100 * acc))
    return acc


def validation_dataloader(model_path: str, use_net=net.VoNet_origin):
    """ データローダからモデルを評価する """
    use_net.to(v.DEVICE)
    # 学習済みモデルを読み込む
    use_net.load_state_dict(torch.load(model_path))

    eval_dataset = dataset.MyCustomDataset(
        v.TRAIN_IMG_DIR,
        v.TEST_CSV,
        dataset.data_transforms['val']
    )
    test_loader = DataLoader(eval_dataset, batch_size=v.BATCH_SIZE)

    print('Testing...')
    _start = time.time()
    validation_net(test_loader, use_net)
    print('Testing time: {} ms.'.format((time.time() - _start) * 1000))
    print('----------------------------------------------------------------')


def validation_custom(mode, dirname, csvname:str=None, use_net=net.VoNet_origin):
    """
    Args:
        mode (str): モードを指定 ('csv', 'dir', 'single' のいずれか)
        dirname (str): 評価用画像のあるディレクトリ名を指定
        csvname (str): CSVモードの場合はCSVファイル名を指定 (dirnameの直下, 拡張子なし)
        use_net (nn.Module): モデルを指定
    Returns:
        None
    """
    use_net.to(v.DEVICE)
    use_net.load_state_dict(torch.load(v.MODEL_PATH))

    if mode == 'csv':
        eval_dataset = dataset.MyCustomDataset(
            dirname, dirname + '/' + csvname + '.csv', dataset.data_transforms['val']
        )
        test_loader = DataLoader(eval_dataset, batch_size=v.BATCH_SIZE)

        print('Testing for {}/{} ...'.format(dirname, csvname))
        validation_net(test_loader, use_net=use_net)
        print('----------------------------------------------------------------')
    elif mode == 'dir':
        dir_imgs = sorted(glob.glob('{}/*.jpg'.format(dirname)))
        for path in dir_imgs:
            image = Image.open(path)
            image = dataset.data_transforms['val'](image).unsqueeze(0).to(v.DEVICE)
            result = use_net(image)
            _, pred = torch.max(result, 1)
            print('{}: {}'.format(path, pred[0].item()))
    elif mode == 'single':
        image = Image.open(dirname)
        image = dataset.data_transforms['val'](image).unsqueeze(0).to(v.DEVICE)
        result = use_net(image)
        _, pred = torch.max(result, 1)
        print('{}: {}'.format(dirname, pred[0].item()))
    else:
        print("mode: 'csv', 'single' or 'dir'.")


def validation_custom_dataset(csvname:str='labels.csv', use_net=net.VoNet_origin):
    """ CUSTOM_VAL_PATHSで指定したパス下にあるデータセットを評価 """
    use_net.to(v.DEVICE)
    use_net.load_state_dict(torch.load(v.MODEL_PATH))
    custom_csv_list, accuracy_list, predicted_list = [], [], []

    for i in range(len(v.CUSTOM_VAL_PATHS)):
        custom_csv_list.append('{}/{}'.format(v.CUSTOM_VAL_PATHS[i], csvname))
        eval_dataset = dataset.MyCustomDataset(
            v.CUSTOM_VAL_PATHS[i],
            custom_csv_list[i],
            dataset.data_transforms['val']
        )
        test_loader = DataLoader(eval_dataset)

        print('Evaluation for {}...'.format(v.CUSTOM_VAL_PATHS[i]))
        accuracy_list.append(validation_net(test_loader, use_net=use_net) * 100)

        data_frame = pd.read_csv(custom_csv_list[i], usecols=[1], header=0)
        img_paths  = data_frame.path.tolist()
        preds      = {}
        for path in img_paths:
            image = Image.open('{}/{}.jpg'.format(v.CUSTOM_VAL_PATHS[i], path))
            image = dataset.data_transforms['val'](image).unsqueeze(0).to(v.DEVICE)
            result = use_net(image)
            _, pred = torch.max(result, 1)
            preds[path] = pred[0].item()
        predicted_list.append(preds)

    _result_path = v.CUSTOM_VAL_OUTPUT_PATH + '.txt'

    with open(_result_path, 'w') as f_stream:
        print('# Result of Custom Dataset Validation', file=f_stream)
        for i in range(len(v.CUSTOM_VAL_PATHS)):
            print('\n----------------------------------------------------------------',
                  file=f_stream)
            print('## No.{}: {}\n'.format(i+1, custom_csv_list[i]), file=f_stream)
            print('Accuracy: %2.2f %%\n' % (accuracy_list[i]), file=f_stream)
            print('Details: ', file=f_stream)
            for key, value in predicted_list[i].items():
                print('{}: {}'.format(key, value), file=f_stream)
    print('Result saved on: ' + _result_path + '. \nFinished.')
