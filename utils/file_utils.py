from typing import Dict, TypeVar, Union

import numpy
import scipy.io as sio
import torch


T = TypeVar('T')


def read_data(file_dir: str) -> Dict[str, torch.Tensor]:
    """ 从指定位置读取 .mat文件, 其中存储的数据是numpy类型

        :param file_dir: 需要读取的文件的位置
        :return: 读取出的数据
        """
    mat = sio.loadmat(file_dir)

    data = dict()
    for key in mat.keys():
        if not key.startswith('__') and not key.endswith('__'):
            data[key] = torch.from_numpy(mat[key]).type(torch.float32)

    return data


def save_data(file_dir: str, data: Dict[str, torch.Tensor]):
    """ 存储数据到指定位置

    :param file_dir: 存储文件的位置
    :param data: 需要存储的数据
    """
    for key in data.keys():
        if data[key].is_cuda:
            data[key] = data[key].cpu().detach_().numpy()

    sio.savemat(file_dir, data)
