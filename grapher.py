# -*- coding: utf-8 -*-
""" grapher.py
数値リストからグラフの描画、書き出しを行う
Usage:
    model/{DATE_STR} に改行をディリミタとしたデータファイル loss-{DATE_STR}.txt を置いて実行
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# 
loss_1224 = ([1.859,  1.614,  1.604,  1.595,  1.439,  0.6534, 0.4613,  0.3803, 0.3257,  0.2978,
              0.2595, 0.2354, 0.2298, 0.2213, 0.1956, 0.1816, 0.1664,  0.1658, 0.1609,  0.1514,
              0.1308, 0.1214, 0.1159, 0.1211, 0.1111, 0.1093, 0.09507, 0.1047, 0.09875, 0.08984])

def grapher(losses:list, acc_list:list, date:str):
    """ グラフを描画してpng, svgに書き出す """
    epochs = range(1, len(losses)+1)

    plt.plot(epochs, losses, label='Training loss')
    plt.xlabel('Epochs')
    plt.ylabel('Losses')
    plt.ylim(0, 10.0)
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(5)) # epoch軸を5刻みに
    # plt.legend()
    plt.savefig('model/{0}/loss-{0}.svg'.format(date))
    plt.savefig('model/{0}/loss-{0}.png'.format(date))
    plt.show()

    plt.plot(epochs, acc_list, label='Training acc')
    plt.xlabel('Epochs')
    plt.ylabel('acc')
    plt.ylim(0, 1.0)
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(5)) # epoch軸を5刻みに
    # plt.legend()
    plt.savefig('model/{0}/acc(val)-{0}.svg'.format(date))
    plt.savefig('model/{0}/acc(val)-{0}.png'.format(date))
    plt.show()


def losslist(date:str):
    """ 数値データの羅列ファイルを読み込みリストを返す """
    filename = 'model/'+date+'/loss-'+date+'.txt'
    loss = 0.0
    losses = []
    with open(filename) as f_stream:
        for line in f_stream:
            loss = float(line)
            losses.append(loss)
    return losses


def acclist(date:str):
    """ 数値データの羅列ファイルを読み込みリストを返す """
    filename = 'model/'+date+'/acc(val)-'+date+'.txt'
    acc = 0.0
    acc_list = []
    with open(filename) as f_stream:
        for line in f_stream:
            acc = float(line)
            acc_list.append(acc)
    return acc_list