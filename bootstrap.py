import argparse
import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

import deep_hsi_prior
from models.band_selection import band_recombination, band_recovery
from utils.data_utils import print_image, min_max_normalize
from utils.file_utils import read_data
from utils.model_utils import model_build


def parse_args():
    parser = argparse.ArgumentParser(description='deep_hs_prior')

    parser.add_argument('--net', dest='net', default='unet', type=str)
    parser.add_argument('--mode', dest='mode', default='base', type=str)
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


if __name__ == '__main__':

    args = parse_args()

    # 读取需要降噪的数据
    file_name = './data/data.pth'
    data_dict = read_data(file_name)
    image = data_dict['image'].cuda()
    decrease_image = data_dict['noise_image'].cuda()
    print_image([image, decrease_image], title='origin image')

    if args.mode == 'base' or args.mode == 'red':
        date = datetime.datetime.now().strftime('%Y-%m-%d.%H-%M-%S')
        writer = SummaryWriter('./logs/denoising_' + args.net + '_' + args.mode + '/' + date)

        net = model_build(args, image)
        output, output_avg = deep_hsi_prior.func(args, image, decrease_image, net, args.mode, writer=writer)
        writer.close()

    elif args.mode == 'band':
        group, recombination_image = band_recombination(decrease_image, group_size=args.group_size)  # 将图像按结构相似度分组
        print(group)

        date = datetime.datetime.now().strftime('%Y-%m-%d.%H-%M-%S')
        log_dir = './logs/denoising_' + args.net + '_' + args.mode + '/'

        output = []
        output_avg = []
        for index, (image_indices, decrease_subimage) in enumerate(zip(group, recombination_image)):
            print('group: [' + str(index + 1) + '/' + str(len(group)) + ']')

            date = datetime.datetime.now().strftime('%Y-%m-%d.%H-%M-%S')
            writer = SummaryWriter(log_dir + date)

            net = model_build(args, decrease_subimage)
            origin_subimage = torch.index_select(image, dim=0, index=torch.tensor(image_indices).cuda())
            out, out_avg = deep_hsi_prior.func(args, origin_subimage, decrease_subimage, net, mode='red', writer=writer)
            output.append(out)
            output_avg.append(out_avg)

            writer.close()

        output = band_recovery(group, output).cuda()
        output_avg = band_recovery(group, output_avg).cuda()

    else:
        raise ValueError('The input parameter --mode is incorrect, you need to choose between base, red and band')

    output = min_max_normalize(output.detach())
    output_avg = min_max_normalize(output_avg.detach())
    print_image([output, output_avg])
