import torch


def noise_generator(method, shape: list) -> torch.Tensor:
    """ 生成一个指定大小的噪声张量

    :param method: 需要生成张量的维度
    :param shape: 张量的形状
    :return: 返回指定大小的噪声张量
    """
    if method == '2D':
        shape.insert(0, 1)
    elif method == '3D':
        shape.insert(0, 1)
        shape.insert(0, 1)
    else:
        assert False

    noise_tensor = torch.zeros(shape)
    noise_tensor.uniform_()
    noise_tensor *= 0.1

    return noise_tensor
