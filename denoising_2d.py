import datetime
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from models.unet2D import UNet
from utils.data_utils import print_image, psnr, min_max_normalize
from utils.model_utils import noise_generator

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def func(args, image, decrease_image):
    data_type = torch.cuda.FloatTensor

    reg_noise_std = args.reg_noise_std
    learning_rate = args.learning_rate
    exp_weight = args.exp_weight
    show_every = args.show_every
    num_iter = args.num_iter

    net = UNet(image.shape[0],
               image.shape[0],
               num_channels_up=[args.up_channel] * 5,
               num_channels_down=[args.down_channel] * 5,
               num_channel_skip=args.skip_channel,
               kernel_size_up=3,
               kernel_size_down=3,
               kernel_size_skip=3,
               downsample_mode=args.downsample_mode,
               upsample_mode=args.upsample_mode,
               need1x1_up=False,
               need_sigmoid=False,
               need_bias=True,
               pad='reflection',
               activate='LeakyReLU').type(data_type)
    device = torch.device('cuda')
    net.to(device)
    # print(net)
    print('module running in: ', device)

    # 计算模型参数
    s = sum([np.prod(list(p.size())) for p in net.parameters()])
    print('number of params: ', s)

    # 拓展数据的维度
    # from [191, 256, 256] to [1, 191, 256, 256]
    decrease_image = decrease_image[None, :].cuda()
    # generates a noise tensor of a specified size
    net_input = noise_generator('2D', list(image.shape)).type(data_type).detach()
    net_input_saved = net_input.detach().clone()  # clone the noise tensor without grad
    noise = net_input.detach().clone()  # clone twice
    out_avg = None  # 上次迭代的输出

    date = datetime.datetime.now().strftime('%Y-%m-%d.%H-%M-%S')
    writer = SummaryWriter('./logs/denoising/' + date)  # the location where the data record is saved

    criterion = torch.nn.MSELoss().type(data_type)                    # 损失函数
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)  # 优化器

    print('start iteration...')
    start_time = time.time()

    for i in range(num_iter + 1):
        optimizer.zero_grad()

        net_input = net_input_saved + (noise.normal_() * reg_noise_std)

        out = net(net_input)  # the result of a network iteration

        # smoothing
        if out_avg is None:
            out_avg = out.detach()
        else:
            out_avg = out_avg * exp_weight + out.detach() * (1 - exp_weight)

        total_loss = criterion(out, decrease_image).to(device)  # calculate the loss value of the loss function
        # print(total_loss)
        total_loss.backward()                                   # back propagation gradient calculation
        optimizer.step()

        # print(decrease_image.shape, out.squeeze().shape)
        psnr_noisy = psnr(decrease_image.squeeze(), out.squeeze())
        psnr_gt = psnr(image, out.squeeze())
        psnr_gt_sm = psnr(image, out_avg.squeeze())

        writer.add_scalar('compare with de', psnr_noisy, i)
        writer.add_scalar('compare with gt', psnr_gt, i)
        writer.add_scalar('compare with gt_sm', psnr_gt_sm, i)

        # backtracking
        if i % show_every == 0:
            msg = 'iteration times: [' + str(i) + '/' + str(num_iter) + ']'
            print(msg)
            out = torch.clamp(out, 0, 1)
            out_avg = torch.clamp(out_avg, 0, 1)

            out_normalize = min_max_normalize(out.squeeze().detach())
            out_avg_normalize = min_max_normalize(out_avg.squeeze().detach())
            print_image([out_normalize, out_avg_normalize], title=msg)
            #
            # if psnr_noisy - psnr_noisy_last < -5:  # model produced an overfit
            #     for new_param, net_param in zip(last_net, net.parameters()):
            #         net_param.detach().copy_(new_param.cuda())  # copy the paras from saved to model
            #
            #     return total_loss * 0  # clean the loss
            # else:  # psnr volatility is still within expectations
            #     last_net = [x.detach().cpu() for x in net.parameters()]  # save the parameters in the model
            #     psnr_noisy_last = psnr_noisy  # renew psnr_noisy_last

    writer.close()
    end_time = time.time()
    print('cost time', end_time - start_time, 's')
    print_image([image, decrease_image.squeeze()], title='origin image')
