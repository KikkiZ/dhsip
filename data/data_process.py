import skimage.io
import torch


def raw_data_process(origin_path: str, target_path: str):
    """ 将原始 .tif 数据转为 .pth

    :param origin_path: 原始数据地址
    :param target_path: 存储数据地址
    """
    image_data = skimage.io.imread(origin_path)  # 读取原始数据 -> ndarray
    image_tensor = torch.from_numpy(image_data)  # ndarray -> tensor
    print(image_tensor.shape)
    torch.save({'image': image_tensor}, target_path)


def data_cut(data_path: str,
             target_path: str,
             start_position: tuple[int, int],
             cut_size: tuple[int, int]):
    """ 切割指定大小的 .pth数据

    :param data_path: 需要切割的数据的地址
    :param target_path: 存储数据的地址
    :param start_position: 切割的起始点
    :param cut_size: 切割数据的大小
    """
    pth = torch.load(data_path)

    data = pth['image']
    end_posit = [start_position[0] + cut_size[0], start_position[1] + cut_size[1]]
    cut_data = data[:, start_position[0]:end_posit[0], start_position[1]:end_posit[1]]
    print(cut_data.shape)
    torch.save({'image': cut_data}, target_path)
