import torch


def band_recombination(data: torch.Tensor):
    if len(data.shape) == 3:
        band = data.shape[0]
        laplace = laplace_matrix(data)

    else:
        assert False


_sigma_s = 0.5
_sigma_r = 0.5


def adjacent_matrix(data: torch.Tensor) -> torch.Tensor:
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
        assert False

    return torch.exp(delta) * torch.exp(pixel_diff)


def diag_matrix(data: torch.Tensor) -> torch.Tensor:
    kernel = torch.ones(1, 1, 3, 3).cuda()
    kernel[0][0][1][1] = 0

    if len(data.shape) == 2:
        width, height = data.shape
    elif len(data.shape) == 3:
        _, width, height = data.shape
    else:
        assert False

    maps = torch.ones(1, width, height).cuda()
    diag = torch.nn.functional.conv2d(maps, kernel, stride=1, padding=1)
    diag = torch.reshape(diag, (-1,))

    return torch.diag(diag)


def laplace_matrix(data: torch.Tensor):
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
        assert False

    return diag - adjacent  # 返回拉普拉斯矩阵
