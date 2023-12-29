import argparse
import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

import deep_hsi_prior
from models.band_selection import band_recombination, band_recovery
from models.resnet import ResNet
from models.unet2D import UNet
from utils.data_utils import print_image, min_max_normalize
from utils.file_utils import read_data


def parse_args():
    parser = argparse.ArgumentParser(description='deep_hs_prior')

    parser.add_argument('--net', dest='net', default='default', type=str)
    parser.add_argument('--num_iter', dest='num_iter', default=3000, type=int)
    parser.add_argument('--reg_noise_std', dest='reg_noise_std', default=0.03, type=float)
    parser.add_argument('--show_every', dest='show_every', default=50, type=int)
    parser.add_argument('--exp_weight', dest='exp_weight', default=0.99, type=float)
    parser.add_argument('--lr', dest='learning_rate', default=0.01, type=float)
    parser.add_argument('--skip_channel', dest='skip_channel', default=4, type=int)
    parser.add_argument('--up_channel', dest='up_channel', nargs='+', default=[128, 128, 128, 128, 128], type=int)
    parser.add_argument('--down_channel', dest='down_channel', nargs='+', default=[128, 128, 128, 128, 128], type=int)
    parser.add_argument('--upsample_mode', dest='upsample_mode', default='bilinear', type=str)
    parser.add_argument('--downsample_mode', dest='downsample_mode', default='stride', type=str)
    parser.add_argument('--group_size', dest='group_size', default=5, type=int)

    return parser.parse_args()


def net_generation(args, image):

    if args.net in ['base', 'red', 'band']:
        net = UNet(image.shape[0],
                   image.shape[0],
                   num_channels_up=args.up_channel,
                   num_channels_down=args.down_channel,
                   num_channel_skip=image.shape[0],
                   kernel_size_up=3,
                   kernel_size_down=3,
                   kernel_size_skip=3,
                   upsample_mode=args.upsample_mode,
                   need1x1_up=False,
                   need_sigmoid=False,
                   need_bias=True,
                   pad='reflection',
                   activate='LeakyReLU')
    elif args.net == 'res':
        net = ResNet(image.shape[0],
                     image.shape[0],
                     num_channels_in=args.up_channel,
                     num_channels_out=args.down_channel,
                     kernel_size=3,
                     activate='LeakyReLU',
                     need_bias=True,
                     pad='reflection')
    else:
        raise ValueError('The input parameter is incorrect, you need to choose between base, red and band')

    return net


if __name__ == '__main__':

    args = parse_args()

    # 读取需要降噪的数据
    file_name = './data/data.pth'
    data_dict = read_data(file_name)
    image = data_dict['image'].cuda()
    decrease_image = data_dict['noise_image'].cuda()
    print_image([image, decrease_image], title='origin image')

    if args.net == 'base':
        date = datetime.datetime.now().strftime('%Y-%m-%d.%H-%M-%S')
        writer = SummaryWriter('./logs/denoising/' + date)

        net = net_generation(args, image)
        output, output_avg = deep_hsi_prior.func(args, image, decrease_image, net, args.net, writer=writer)
        writer.close()

    elif args.net == 'red':
        date = datetime.datetime.now().strftime('%Y-%m-%d.%H-%M-%S')
        writer = SummaryWriter('./logs/denoising_red/' + date)

        net = net_generation(args, image)
        output, output_avg = deep_hsi_prior.func(args, image, decrease_image, net, args.net, writer=writer)
        writer.close()

    elif args.net == 'res':
        date = datetime.datetime.now().strftime('%Y-%m-%d.%H-%M-%S')
        writer = SummaryWriter('./logs/denoising_res/' + date)

        net = net_generation(args, image)
        output, output_avg = deep_hsi_prior.func(args, image, decrease_image, net, args.net, writer=writer)
        writer.close()

    elif args.net == 'band':
        group, recombination_image = band_recombination(decrease_image, group_size=args.group_size)  # 将图像按结构相似度分组
        print(group)

        date = datetime.datetime.now().strftime('%Y-%m-%d.%H-%M-%S')
        log_dir = './logs/denoising_band/' + date + '/'

        output = []
        output_avg = []
        for index, (image_indices, decrease_subimage) in enumerate(zip(group, recombination_image)):
            print('group: [' + str(index + 1) + '/' + str(len(group)) + ']')

            date = datetime.datetime.now().strftime('%Y-%m-%d.%H-%M-%S')
            writer = SummaryWriter(log_dir + date)

            net = net_generation(args, decrease_subimage)
            origin_subimage = torch.index_select(image, dim=0, index=torch.tensor(image_indices).cuda())
            out, out_avg = deep_hsi_prior.func(args, origin_subimage, decrease_subimage, net, mode='red', writer=writer)
            output.append(out)
            output_avg.append(out_avg)

            writer.close()

        output = band_recovery(group, output).cuda()
        output_avg = band_recovery(group, output_avg).cuda()

    else:
        raise ValueError('The input parameter is incorrect, you need to choose between base, red and band')

    output = min_max_normalize(output.detach())
    output_avg = min_max_normalize(output_avg.detach())
    print_image([output, output_avg])
