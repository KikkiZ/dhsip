import time
import datetime

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from models.non_local_means import non_local_means
from models.unet2D import UNet
from models.band_selection import band_recovery
from utils.data_utils import min_max_normalize, print_image, psnr
from utils.file_utils import read_data, save_data
from utils.model_utils import noise_generator

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
data_type = torch.cuda.FloatTensor


def func(args,
         image: torch.Tensor,
         decrease_image: list[torch.Tensor],
         group: list[list[int]]):
    # reg_noise_std = args.reg_noise_std
    # learning_rate = args.learning_rate
    # exp_weight = args.exp_weight
    # show_every = args.show_every
    # num_iter = args.num_iter
    mu = 0.5
    beta = 0.5

    reg_noise_std = 0.03
    learning_rate = 0.01
    exp_weight = 0.99
    show_every = 200
    num_iter = 4000
    print(len(group), len(decrease_image))

    image = image.cuda()

    date = datetime.datetime.now().strftime('%Y-%m-%d.%H-%M-%S')
    log_dir = './logs/denoising_band/' + date + '/'

    output = []
    for index, (image_indices, decrease_subimage) in enumerate(zip(group, decrease_image)):
        msg = 'epoch: [' + str(index + 1) + '/' + str(len(group)) + ']'
        print(msg)

        net = UNet(decrease_subimage.shape[0],
                   decrease_subimage.shape[0],
                   num_channels_up=[16, 32, 64, 128, 128],
                   num_channels_down=[16, 32, 64, 128, 128],
                   num_channel_skip=4,
                   kernel_size_up=3,
                   kernel_size_down=3,
                   kernel_size_skip=3,
                   upsample_mode='bilinear',
                   need1x1_up=False,
                   need_sigmoid=False,
                   need_bias=True,
                   pad='reflection',
                   activate='LeakyReLU').type(data_type)
        device = torch.device('cuda')
        net.to(device)

        s = sum([np.prod(list(p.size())) for p in net.parameters()])
        print('number of params: ', s)

        origin_subimage = torch.index_select(image, dim=0, index=torch.tensor(image_indices).cuda())
        net_input = noise_generator('2D', list(decrease_subimage.shape)).type(data_type).detach()
        net_input_saved = net_input.detach().clone()  # clone the noise tensor without grad
        noise = net_input.detach().clone()  # clone twice

        # 该值代表损失函数约束的拉格朗日乘子张量
        lagrange_multiplier = torch.zeros(origin_subimage.shape).type(data_type)[None, :].cuda()
        # 该值初始以退化图像作为基准值, 后续会不断更新该值
        benchmark_image = decrease_subimage.clone().type(data_type)[None, :].cuda()
        temp_benchmark = benchmark_image.clone()  # 基准值的副本, 用于中间计算

        out_avg = None  # 上次迭代的输出
        out = None

        decrease_subimage = decrease_subimage[None, :].cuda()

        criterion = torch.nn.MSELoss().type(data_type)  # 损失函数
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)  # 优化器

        date = datetime.datetime.now().strftime('%Y-%m-%d.%H-%M-%S')
        writer = SummaryWriter(log_dir + date)

        start_time = time.time()

        # 运行基准函数, 更新基准值
        benchmark_image = non_local_means(benchmark_image.clone().squeeze(), 3)

        for i in range(num_iter + 1):
            optimizer.zero_grad()

            net_input = net_input_saved + (noise.normal_() * reg_noise_std)
            out = net(net_input)

            temp = benchmark_image - lagrange_multiplier
            loss_net = criterion(out, decrease_subimage)  # 网络的loss
            loss_red = criterion(out, temp.detach_())  # RED's loss

            total_loss = loss_net + mu * loss_red

            total_loss.backward()
            optimizer.step()  # update the parameters of the model

            if i % 25 == 0:
                temp_benchmark = non_local_means(benchmark_image.clone().squeeze(), 3)

            benchmark_image = 1 / (beta + mu) * (beta * temp_benchmark + mu * (out + lagrange_multiplier))
            lagrange_multiplier = lagrange_multiplier + out - benchmark_image

            if out_avg is None:
                out_avg = out.detach()
            else:
                out_avg = out_avg * exp_weight + out.detach() * (1 - exp_weight)

            out = min_max_normalize(out.squeeze())
            out_avg = min_max_normalize(out_avg.squeeze())

            psnr_gt = psnr(origin_subimage, out.squeeze())
            psnr_gt_sm = psnr(origin_subimage, out_avg.squeeze())

            writer.add_scalar('cmp with gt', psnr_gt, i)
            writer.add_scalar('cmp with gt_sm', psnr_gt_sm, i)

            if i % show_every == 0:
                msg = 'iteration times: [' + str(i) + '/' + str(num_iter) + ']'
                print(msg)
            #     out = torch.clamp(out, 0, 1)
            #     out_avg = torch.clamp(out_avg, 0, 1)
            #
            #     out_normalize = min_max_normalize(out.squeeze().detach())
            #     out_avg_normalize = min_max_normalize(out_avg.squeeze().detach())
            #     print_image([out_normalize, out_avg_normalize], title=msg, bands=[0, 1, 2])

        writer.close()
        output.append(out.squeeze())
        end_time = time.time()
        print(end_time - start_time)

    # 数据重组
    output = band_recovery(group, output).cuda()
    print(output.shape)
    print_image([output.detach()])
    print(psnr(image, output))
    save_data('./data/output.mat', {'output': output.clone()})
