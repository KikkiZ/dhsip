from typing import Dict

import torch


def read_data(file_dir: str):
    """ 从指定位置读取 .pth文件, 其中存储的数据是torch的张量

        :param file_dir: 需要读取的文件的位置
        :return: 读取出的数据
        """
    pth = torch.load(file_dir)

    return pth


def save_data(file_dir: str, data):
    """ 存储数据到指定位置

    :param file_dir: 存储文件的位置
    :param data: 需要存储的数据
    """
    torch.save(data, file_dir)
