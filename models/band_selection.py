import math

import numpy
import torch


def band_recombination(data: torch.Tensor,
                       group_size: int = 3) -> tuple[list[list[int]], list[torch.Tensor]]:
    """ 将一张高光谱图像按结构相似度进行分组
    输入一张高光谱图像, 图像结构为(C,W,H),
    通过计算出的结构相似度, 将光谱图像按相似度重组

    :param data: 需要重组的高光谱图像
    :param group_size: 重组的图像的大小
    :return: 返回分组信息和重组完的图像
    """
    if len(data.shape) != 3:
        raise ValueError('The input tensor dimension is incorrect')

    data = data.cuda()
    sub_image_size = 16
    band, width, height = data.shape
    if width < sub_image_size * 2 and height < sub_image_size * 2:
        # 当输入数据大小未超过阈值时, 可以直接进行结构相似度计算
        similarity = structure_info(data)

    else:
        num_row = math.ceil(width / sub_image_size)
        num_col = math.ceil(height / sub_image_size)

        # 将输入数据切割为指定大小
        blocks = []
        row_blocks = torch.chunk(data, num_row, dim=1)
        for row_block in row_blocks:
            col_blocks = torch.chunk(row_block, num_col, dim=2)
            blocks.extend(col_blocks)

        print('total number of blocks: ', len(blocks))

        # 计算结构相似度
        similarity = []
        for index, block in enumerate(blocks):
            print('processing progress: [', index + 1, '/', len(blocks), ']')
            info = structure_info(block)
            similarity.append(info)

        similarity = torch.stack(similarity, dim=0)  # 将分组的相似度合并, 通道数为分组数量
        similarity = torch.sum(similarity, dim=0)  # 将相似度数据叠加, 减少通道数

    similarity = similarity.cpu().numpy()
    similarity[similarity == 0] = numpy.inf

    index = 0
    group = []

    loop_count = 1
    loop_num = round(band / group_size)
    while loop_count <= loop_num:
        # 检查当前行是否已被分配
        arr = similarity[index]
        if numpy.all(numpy.isinf(arr)):
            index += 1
            continue

        # 调整最后一组数据的大小
        if loop_count == loop_num:
            group_size = band - (loop_count - 1) * group_size
            print('group size: ', group_size)

        # 筛选剩余波段中最匹配的波段, 处理分配信息
        indices = numpy.argpartition(arr, group_size - 1)[: group_size - 1]
        similarity[:, indices] = numpy.inf  # 纵向清空结构相似性数据
        similarity[indices, :] = numpy.inf  # 横向清空结构相似性数据
        similarity[index, :] = numpy.inf

        indices = indices.tolist()
        indices.extend([index])
        loop_count += 1
        index += 1

        group.append(indices)

    recombination_image = []
    for indices in group:
        image = torch.index_select(data, 0, torch.tensor(indices).cuda())
        recombination_image.append(image)

    return group, recombination_image


def band_recovery(group: list[list[int]], data: list[torch.Tensor]) -> torch.Tensor:
    """ 根据分组信息恢复原始图像
    默认data中的所有张量具有相同的大小(通道数可以不同),
    group中的元素数量与与data中的通道数相同

    :param group: 图像的分组信息
    :param data: 分组的图像
    :return： 返回重组完成的图像
    """
    _, width, height = data[0].shape

    channel = sum([len(item) for item in group])
    recover_data = torch.zeros([channel, width, height])
    for indices, image in zip(group, data):
        for index, origin_index in enumerate(indices):
            recover_data[origin_index] = image[index]

    return recover_data


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
        data = data.cuda()
        laplace = laplace_matrix(data)

        # 由于直接计算内存占用过高, 需要按批次计算
        batch_size = 3000  # 每一批的大小
        num_batch = math.ceil(band ** 2 / batch_size)  # 分批计算的总批次
        indices = torch.triu_indices(band, band, offset=1)  # 生成上三角矩阵的索引, 共两个列表, 分别存有x和y轴
        indices_x = indices[0]  # 获取需要进行计算的第一个张量的索引列表
        indices_y = indices[1]  # 获取第二个张量的索引列表

        structural_similarity = []
        for batch in range(num_batch):
            temp_x = indices_x[batch * batch_size: (batch + 1) * batch_size]
            temp_y = indices_y[batch * batch_size: (batch + 1) * batch_size]
            laplace_x = laplace[temp_x]
            laplace_y = laplace[temp_y]

            # 'ijk,ikl->ijl' 使用了爱因斯坦求和约定来描述操作运算
            # 'ijk,ikl' 表示输入两个输入张量维度的标签, 'ijl'表示输出张量的维度的标签
            # 当标签输入维度标签相同时视为求和, 不同的维度标签被视为进行乘法
            # 该式描述的操作: i维度上进行求和; jk和kl进行矩阵乘, 生成jl大小的矩阵; 最终生成了ijl大小的矩阵
            info = (torch.einsum('ijk,ikl->ijl', laplace_x, laplace_y) -
                    torch.einsum('ijk,ikl->ijl', laplace_y, laplace_x))
            info = torch.norm(info, dim=[1, 2]).pow(2)
            structural_similarity.extend(info.tolist())

        structural_similarity = torch.tensor(structural_similarity)

        upper_triangular = torch.zeros(band, band)
        indices = torch.triu_indices(band, band, offset=1)
        upper_triangular[indices[0], indices[1]] = structural_similarity

        return upper_triangular

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
        diag = diag[None, :]  # 拓展对角矩阵的维度
        diag = diag.expand(band, -1, -1)  # 将对角矩阵拓展到输入数据相同通道数

        adjacent = adjacent_matrix(data)

    elif len(data.shape) == 2:
        diag = diag_matrix(data)
        adjacent = adjacent_matrix(data)

    else:
        raise ValueError('The input tensor dimension is incorrect')

    return diag - adjacent  # 返回拉普拉斯矩阵
