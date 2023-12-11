import torch


def _tensor_repeat(inputs, x, y):
    inputs = inputs.repeat(x, y, 1)
    inputs = torch.transpose(inputs, 0, 2)
    inputs = torch.transpose(inputs, 1, 2)
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

    min_tensor = torch.min(inputs, dim=dim).values
    min_tensor = torch.min(min_tensor, dim=dim).values
    min_tensor = _tensor_repeat(min_tensor, x=x, y=y)

    max_tensor = torch.max(inputs, dim=dim).values
    max_tensor = torch.max(max_tensor, dim=dim).values
    max_tensor = _tensor_repeat(max_tensor, x=x, y=y)

    return (inputs - min_tensor) / (max_tensor - min_tensor)


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
