import random
from typing import Iterable, Dict

import torch
from matplotlib import pyplot as plt


def _tensor_repeat(inputs, x, y):
    inputs = inputs.repeat(x, y, 1)
    inputs = inputs.permute(2, 0, 1)
    return inputs


def min_max_normalize(inputs: torch.Tensor) -> torch.Tensor:
    """ 最小-最大归一化
    默认输入数据格式为(W,H)或(C,W,H)

    :param inputs: 输入需要归一化的数据
    :return: 返回归一化后的结果
    """
    if len(inputs.shape) == 2:
        dim = 0
        x, y = inputs.shape

    elif len(inputs.shape) == 3:
        dim = 1
        _, x, y = inputs.shape

    else:
        raise ValueError('The input tensor dimension is incorrect')

    if not inputs.is_floating_point():
        inputs = inputs.to(torch.float)

    min_tensor = torch.min(inputs, dim=dim).values
    min_tensor = torch.min(min_tensor, dim=dim).values
    min_tensor = _tensor_repeat(min_tensor, x=x, y=y)

    max_tensor = torch.max(inputs, dim=dim).values
    max_tensor = torch.max(max_tensor, dim=dim).values
    max_tensor = _tensor_repeat(max_tensor, x=x, y=y)

    return (inputs - min_tensor) / (max_tensor - min_tensor)


def add_white_noise(inputs: torch.Tensor, noise_level: int) -> torch.Tensor:
    """ 给图像添加高斯白噪声

    :param inputs: 需要添加噪声的图像
    :param noise_level: 噪声水平
    :return: 返回添加了噪声的图像
    """
    white_noise = torch.randn(inputs.shape) * (noise_level / 255)

    image_noise = inputs + white_noise
    image_noise = torch.clamp(image_noise, 0, 1)

    return image_noise


def add_white_noise_by_band(inputs: torch.Tensor,
                            noise_levels: Dict[int, float]) -> torch.Tensor:
    """ 给图像添加不同水平的高斯白噪声

    :param inputs: 需要添加噪声的图像
    :param noise_levels: 噪声水平以及对应等级出现的概率
    :return: 返回添加了噪声的图像
    """
    if sum(noise_levels.values()) != 1:
        raise ValueError('The sum of noise levels probabilities is not one')

    if len(inputs.shape) != 3:
        raise ValueError('The input tensor dimension must be 3')

    output = inputs.detach().clone()
    channel, width, height = output.shape
    band_count = [round(channel * temp) for temp in noise_levels.values()]
    rand_list = random.sample(range(0, channel), channel)

    for index, level in enumerate(noise_levels.keys()):
        band_list = rand_list[:band_count[index]]
        rand_list = rand_list[band_count[index]:]

        for band in band_list:
            white_noise = torch.randn(width, height) * (level / 255)

            output[band] = output[band] + white_noise
            output[band] = torch.clamp(output[band], 0, 1)

    return output


def psnr(original: torch.Tensor, compressed: torch.Tensor) -> torch.Tensor:
    """ 峰值信噪比
    使用torch重写, 用于快速计算psnr

    :param original: 原始图像
    :param compressed: 比较图像
    :return: 返回两张图像的psnr
    """
    if not original.shape == compressed.shape:
        print(original.shape)
        print(compressed.shape)
        raise ValueError('Input must have the same dimensions.')

    if original.dtype != compressed.dtype:
        raise TypeError("Inputs have mismatched dtype. Set both tensors to be of the same type.")

    true_max = torch.max(original)
    true_min = torch.min(original)
    if true_max > 1 or true_min < 0:
        raise ValueError("image_true has intensity values outside the range expected "
                         "for its data type. Please manually specify the data_range.")

    err = torch.mean((original - compressed) ** 2)  # 均方误差
    return (10 * torch.log10(1 / err)).item()


def mpsnr(original: torch.Tensor, compressed: torch.Tensor) -> torch.Tensor:
    """ 平均峰值信噪比
    使用torch重写, 用于快速计算mpsnr

    :param original: 原始图像
    :param compressed: 比较图像
    :return: 返回两张图像的mpsnr
    """
    err = torch.mean((original - compressed) ** 2, dim=[1, 2])
    err = 10 * torch.log10(1 / err)
    return torch.mean(err).item()


def print_image(images: list[torch.Tensor],
                bands: list[int] = None,
                title: str = None):
    """ 输出一组任意长度的(高光谱)图像

    :param images: 需要输出的一组图像
    :param title: 图像的标题
    :param bands: 输出图像的波段
    """
    if bands is None:
        bands = [56, 26, 16]

    f, axs = plt.subplots(nrows=1,
                          ncols=len(images),
                          sharey=True,
                          figsize=(5 if len(images) == 1 else 4 * len(images), 5))

    if title is not None:
        f.suptitle(title)

    if isinstance(axs, Iterable):
        for index, ax in enumerate(axs):
            var = images[index]
            ax.imshow(torch.stack((var[bands[0], :, :], var[bands[1], :, :], var[bands[2], :, :]), 2).cpu())
    else:
        var = images[0]
        axs.imshow(torch.stack((var[bands[0], :, :], var[bands[1], :, :], var[bands[2], :, :]), 2).cpu())

    plt.show()
