import time
from typing import Tuple

import numpy
import torch
from torch.utils.tensorboard import SummaryWriter

from models.non_local_means import non_local_means
from utils.data_utils import min_max_normalize, print_image, psnr
from utils.model_utils import noise_generator

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
data_type = torch.cuda.FloatTensor
device = torch.device('cuda')


def func(args,
         image: torch.Tensor,
         decrease_image: torch.Tensor,
         net: torch.nn.Module,
         mode: str = 'base',
         writer: SummaryWriter = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """ 训练模型

    :param args: 程序输入的参数
    :param image: 干净图像
    :param decrease_image:退化图像
    :param net: 网络模型
    :param mode: 网络运行的模式
    :param writer: 网络数据记录
    :return: 返回去噪图像和平滑后的去噪图像组成的元组
    """

    reg_noise_std = args.reg_noise_std  # 扰动噪声张量的常量
    learning_rate = args.learning_rate  # 学习率
    exp_weight = args.exp_weight        # 平滑参数, 将网络输出与过往输出值
    show_every = args.show_every
    num_iter = args.num_iter            # 模型迭代次数
    mu = 0.5                            # ADMM参数，希腊字母μ
    beta = 0.5                          # 正则化参数

    net = net.type(data_type)
    net.to(device)
    print('module running in: ', device)

    s = sum([numpy.prod(list(p.size())) for p in net.parameters()])  # 计算模型参数
    print('number of params: ', s)

    # 该值代表损失函数约束的拉格朗日乘子张量
    lagrange_multiplier = torch.zeros(image.shape).type(data_type)[None, :].cuda()
    # 该值初始以退化图像作为基准值, 后续会不断更新该值
    benchmark_image = decrease_image.clone().type(data_type)[None, :].cuda()
    temp_benchmark = benchmark_image.clone()  # 基准值的副本, 用于中间计算
    out = None
    out_avg = None                            # 用于记录平滑输出的累计值

    # 拓展数据的维度
    decrease_image = decrease_image[None, :].cuda()
    # 自动生成一个指定维度的噪声图像
    net_input = noise_generator('2D', list(image.shape)).type(data_type).detach()
    noise = net_input.clone().detach_()

    print('start iteration...')
    start_time = time.time()

    criterion = torch.nn.MSELoss().type(data_type)                    # 损失函数
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)  # 优化器

    # 运行基准函数, 更新基准值
    benchmark_image = non_local_means(benchmark_image.clone().squeeze(), 3)

    for i in range(num_iter + 1):
        optimizer.zero_grad()  # 清空梯度

        inputs = net_input + (noise.normal_() * reg_noise_std)  # 扰动每一轮的网络输入噪声
        out = net(inputs)
        out = min_max_normalize(out.squeeze())[None, :]

        if mode == 'base':
            total_loss = criterion(out, decrease_image)
        elif mode == 'red':
            temp = benchmark_image - lagrange_multiplier
            loss_net = criterion(out, decrease_image)  # 网络的loss
            loss_red = criterion(out, temp.detach_())  # RED's loss

            total_loss = loss_net + mu * loss_red

            benchmark_image = 1 / (beta + mu) * (beta * temp_benchmark + mu * (out + lagrange_multiplier))
            lagrange_multiplier = lagrange_multiplier + out - benchmark_image
        else:
            raise TypeError('incorrect mode value')

        total_loss.backward()
        optimizer.step()  # update the parameters of the model

        # 当模式为red时, 模型每迭代一定次数更新一次基准值
        if i % 25 == 0 and mode == 'red':
            temp_benchmark = non_local_means(benchmark_image.clone().squeeze(), 3)

            msg = 'benchmark: [' + str(i) + '/' + str(num_iter) + ']'
            benchmark_image_normalize = min_max_normalize(benchmark_image.squeeze().detach())
            lagrange_multiplier_normalize = min_max_normalize(lagrange_multiplier.squeeze().detach())
            print_image([benchmark_image_normalize, lagrange_multiplier_normalize], title=msg)

        if out_avg is None:
            out_avg = out.detach()
        else:
            out_avg = out_avg * exp_weight + out.detach() * (1 - exp_weight)

        # 当需要记录数据时才计算psnr
        if writer is not None:
            psnr_noisy = psnr(decrease_image.squeeze(), out.squeeze())
            psnr_gt = psnr(image, out.squeeze())
            psnr_gt_sm = psnr(image, out_avg.squeeze())

            writer.add_scalar('cmp with de', psnr_noisy, i)
            writer.add_scalar('cmp with gt', psnr_gt, i)
            writer.add_scalar('cmp with gt_sm', psnr_gt_sm, i)

        if i % show_every == 0:
            msg = 'iteration times: [' + str(i) + '/' + str(num_iter) + ']'
            print(msg)

            # 只有当模式不为波段重组时才会输出图像
            if args.mode != 'band':
                out_normalize = min_max_normalize(out.squeeze().detach())
                out_avg_normalize = min_max_normalize(out_avg.squeeze().detach())
                print_image([out_normalize, out_avg_normalize], title=msg)

    end_time = time.time()
    print('cost time: ', end_time - start_time, 's')

    return out.squeeze(), out_avg.squeeze()
