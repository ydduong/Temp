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
        count = 0
        while True:
            count += 1
            if count % 2 != 0:
                _r.readline()
                _r.readline()
                continue
            # if count > 300:
            #     break

            line = _r.readline()
            if line == "":
                break

            _line = line.strip()
            _data = _line.split(',')

            if _data[-1] == 'train':
                _train_acc.append(float(_data[-2]))
                _train_loss.append(float(_data[-3]))
            if _data[-1] == 'val':
                _val_acc.append(float(_data[-2]))
                _val_loss.append(float(_data[-3]))

            line = _r.readline()
            if line == "":
                print("kk")
                break

            _line = line.strip()
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
    print(len(_train_loss))

    plt.figure(figsize=(8, 5))
    plt.title('loss value')
    plt.plot(_x, _train_loss, label="train")
    plt.plot(_x, _val_loss, label="val")
    plt.ylim(min(_val_loss) - 0.1, max(_val_loss) + 0.1)
    # plt.ylim(min([min(_train_loss), min(_val_loss)])-0.1, max([max(_train_loss), max(_val_loss)])+0.1)
    plt.legend()

    plt.figure(figsize=(8, 5))
    plt.title('accuracy')
    plt.plot(_x, _train_acc, label="train")
    plt.plot(_x, _val_acc, label="val")
    # plt.ylim(min([min(_train_acc), min(_val_acc)])-10, max([max(_train_acc), max(_val_acc)])+10)
    plt.ylim(min(_val_acc) - 10, max(_val_acc) + 10)
    plt.legend()

    plt.show()


if __name__ == '__main__':
    # model_name = 'Cnn'
    model_name = 'FlowCNNLearningCurve.csv'
    # 训练日志
    # file = os.path.join(args.model_dir, model_name)
    # 学习曲线
    learning_curve(model_name)
