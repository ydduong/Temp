# encoding=utf-8

import os
import sys

import matplotlib.pyplot as plt
import numpy as np


def get_data(_file):
    _is_csv = True if _file[-3:] == 'csv' else False
    if not _is_csv:
        print(f'error: this is not csv file, no table() function. {_file}')
        sys.exit(-1)

    _train_loss, _train_acc, _val_loss, _val_acc = [], [], [], []
    with open(_file, "r", encoding="utf-8") as _r:
        _lines = _r.readlines()
        for _line in _lines:
            _line = _line.strip()
            _data = _line.split(',')

            if _data[-1] == 'train':
                _train_acc.append(float(_data[-2]))
                _train_loss.append(float(_data[-3]))

            if _data[-1] == 'val':
                _val_acc.append(float(_data[-2]))
                _val_loss.append(float(_data[-3]))

    return _train_loss, _train_acc, _val_loss, _val_acc


# 绘制学习曲线
def learning_curve(file_path):
    _train_loss, _train_acc, _val_loss, _val_acc = get_data(file_path)

    _x = np.arange(len(_train_loss))

    plt.figure('loss', figsize=(8, 5))
    plt.plot(_x, _train_loss)
    plt.plot(_x, _val_loss)
    plt.ylim(0, 1)

    plt.figure('acc', figsize=(8, 5))
    plt.plot(_x, _train_acc)
    plt.plot(_x, _val_acc)
    plt.ylim(0, 30)

    plt.show()


if __name__ == '__main__':
    # model_name = 'Cnn'
    model_name = 'DeepCnn'
    # 训练日志
    file = f'{model_name}LearningCurve.csv'
    # 学习曲线
    learning_curve(file)
