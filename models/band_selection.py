import torch


def band_recombination(data):
    band = data.shape[0]

    laplace_matrix = list()
    for index in range(band):
        print(index)

    return


_sigma_s = 0.5
_sigma_r = 0.5


def adjacent_matrix(data: torch.Tensor):

    length = torch.numel(data)
    data = torch.reshape(data, (-1, ))  # 重塑数据形状

    # δ矩阵
    row = torch.arange(length).unsqueeze(1).expand(length, length).cuda()  # 行
    col = torch.arange(length).unsqueeze(0).expand(length, length).cuda()  # 列
    delta = row - col
    delta = delta ** 2
    delta = -delta / (2 * (_sigma_s ** 2))

    row = data.unsqueeze(1).expand(length, length).cuda()
    col = data.unsqueeze(0).expand(length, length).cuda()
    pixel_diff = row - col
    pixel_diff = -(pixel_diff ** 2) / (2 * (_sigma_r ** 2))

    adj_matrix = torch.exp(delta) * torch.exp(pixel_diff)
    return adj_matrix


# TODO 完成diag矩阵计算方法
