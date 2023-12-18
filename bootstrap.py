import argparse
import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

import deep_hsi_prior
from models.band_selection import band_recombination, band_recovery
from utils.data_utils import print_image
from utils.file_utils import read_data, save_data


def parse_args():
    parser = argparse.ArgumentParser(description='deep_hs_prior')

    # TODO 删除不必要的参数
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

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    # 读取需要降噪的数据
    file_name = './data/denoising.pth'
    data_dict = read_data(file_name)
    image = data_dict['image'].cuda()
    decrease_image = data_dict['image_noisy'].cuda()
    print_image([image, decrease_image], title='origin image')

    if args.net == 'base':
        date = datetime.datetime.now().strftime('%Y-%m-%d.%H-%M-%S')
        writer = SummaryWriter('./logs/denoising/' + date)

        output, _ = deep_hsi_prior.func(args, image, decrease_image, args.net, writer=writer)
        writer.close()

    elif args.net == 'red':
        date = datetime.datetime.now().strftime('%Y-%m-%d.%H-%M-%S')
        writer = SummaryWriter('./logs/denoising_red/' + date)

        output, _ = deep_hsi_prior.func(args, image, decrease_image, args.net, writer=writer)
        writer.close()

    elif args.net == 'band':
        group, recombination_image = band_recombination(decrease_image, group_size=5)  # 将图像按结构相似度分组
        print(group)

        date = datetime.datetime.now().strftime('%Y-%m-%d.%H-%M-%S')
        log_dir = './logs/denoising_band/' + date + '/'

        output = []
        output_avg = []
        for index, (image_indices, decrease_subimage) in enumerate(zip(group, recombination_image)):
            print('epoch: [' + str(index + 1) + '/' + str(len(group)) + ']')

            date = datetime.datetime.now().strftime('%Y-%m-%d.%H-%M-%S')
            writer = SummaryWriter(log_dir + date)

            origin_subimage = torch.index_select(image, dim=0, index=torch.tensor(image_indices).cuda())
            out, out_avg = deep_hsi_prior.func(args, origin_subimage, decrease_subimage, mode='red', writer=writer)
            output.append(out)
            output_avg.append(out_avg)

            writer.close()

        output = band_recovery(group, output).cuda()
        output_avg = band_recovery(group, output_avg).cuda()
        print_image([output.detach()])
        save_data('./data/output.pth',
                  {'output': output.detach_().clone(), 'output_avg': output_avg.detach_().clone()})

    else:
        raise ValueError('The input parameter is incorrect, you need to choose between base, red and band')
