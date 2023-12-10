import torch


def structure_info(data: torch.Tensor) -> torch.Tensor:
    """ 计算一组图像间的结构相似度
    默认输入的图像为(C,W,H), 通过拉普拉斯矩阵计算
    该组图像的结构相似度, 并返回一张二维矩阵,行号和
    列号分别代表比较的双方, 对应的值为二者的结构相似度

    :param data: 输入需要分组的图像
    :return: 返回一张二维矩阵
    """
    if len(data.shape) == 3:
        band = data.shape[0]
        laplace = laplace_matrix(data)

        structural_similarity = torch.zeros(band, band)
        for i in range(band):
            for j in range(i + 1, band):
                # @ 为矩阵乘
                i = laplace[i] @ laplace[j] - laplace[j] @ laplace[i]
                structural_similarity[i][j] = torch.norm(i).item() ** 2

        return structural_similarity

    else:
        raise ValueError('The input tensor dimension is incorrect')


_sigma_s = 0.5
_sigma_r = 0.5


def adjacent_matrix(data: torch.Tensor) -> torch.Tensor:
    """ 计算输入图像的邻接矩阵
    可以输入单张(2D)图像或多张图像(3D),
    函数会自动计算, 并返回相同通道数的邻接矩阵

    :param data: 输入需要计算的图像
    :return: 返回图像对应的邻接矩阵
    """
    def _delta(size: int):
        _row = torch.arange(size).unsqueeze(1).expand(size, size).cuda()  # 行
        _col = torch.arange(size).unsqueeze(0).expand(size, size).cuda()  # 列
        matrix = _row - _col
        matrix = matrix ** 2
        matrix = -matrix / (2 * (_sigma_s ** 2))
        return matrix

    if len(data.shape) == 2:
        length = torch.numel(data)
        data = torch.reshape(data, (-1,))  # 重塑数据形状

        delta = _delta(length)
        row = data.unsqueeze(1).expand(length, length).cuda()
        col = data.unsqueeze(0).expand(length, length).cuda()
        pixel_diff = row - col
        pixel_diff = -(pixel_diff ** 2) / (2 * (_sigma_r ** 2))

    elif len(data.shape) == 3:
        channel = data.shape[0]
        length = torch.numel(data[0])
        data = torch.reshape(data, (channel, -1))

        delta = _delta(length)
        delta = delta.expand(channel, -1, -1)

        row = data.unsqueeze(2).expand(channel, length, length).cuda()
        col = data.unsqueeze(1).expand(channel, length, length).cuda()
        pixel_diff = row - col
        pixel_diff = -(pixel_diff ** 2) / (2 * (_sigma_r ** 2))

    else:
        raise ValueError('The input tensor dimension is incorrect')

    return torch.exp(delta) * torch.exp(pixel_diff)


def diag_matrix(data: torch.Tensor) -> torch.Tensor:
    """ 计算图像的度矩阵
    输入一张或一组图像, 默认输入的图像为(C,W,H),
    函数将自动生成一张度矩阵, 并转换为对角矩阵

    :param data: 输入需要计算的图像
    :return: 返回图像对应的度矩阵
    """
    kernel = torch.ones(1, 1, 3, 3).cuda()
    kernel[0][0][1][1] = 0

    if len(data.shape) == 2:
        width, height = data.shape
    elif len(data.shape) == 3:
        _, width, height = data.shape
    else:
        raise ValueError('The input tensor dimension is incorrect')

    maps = torch.ones(1, width, height).cuda()
    diag = torch.nn.functional.conv2d(maps, kernel, stride=1, padding=1)
    diag = torch.reshape(diag, (-1,))

    return torch.diag(diag)


def laplace_matrix(data: torch.Tensor) -> torch.Tensor:
    """ 计算图像的拉普拉斯矩阵
    输入一张或一组图像, 默认输入的图像为(W,H)/(C,W,H),
    函数将分别计算邻接矩阵和度矩阵, 并计算出拉普拉斯矩阵

    :param data: 输入需要计算的图像
    :return: 返回图像对应的拉普拉斯矩阵
    """
    if len(data.shape) == 3:
        band = data.shape[0]

        diag = diag_matrix(data)
        diag = diag[None, :]              # 拓展对角矩阵的维度
        diag = diag.expand(band, -1, -1)  # 将对角矩阵拓展到输入数据相同通道数

        adjacent = adjacent_matrix(data)

    elif len(data.shape) == 2:
        diag = diag_matrix(data)
        adjacent = adjacent_matrix(data)

    else:
        raise ValueError('The input tensor dimension is incorrect')

    return diag - adjacent  # 返回拉普拉斯矩阵
