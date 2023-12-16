import numpy as np
import torch

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
    reg_noise_std = 0.03
    learning_rate = 0.01
    exp_weight = 0.99
    show_every = 200
    num_iter = 3000
    print(len(group), len(decrease_image))

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

        net_input = noise_generator('2D', list(decrease_subimage.shape)).type(data_type).detach()
        net_input_saved = net_input.detach().clone()  # clone the noise tensor without grad
        noise = net_input.detach().clone()  # clone twice
        out_avg = None  # 上次迭代的输出
        out = None

        decrease_subimage = decrease_subimage[None, :].cuda()

        criterion = torch.nn.MSELoss().type(data_type)  # 损失函数
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)  # 优化器

        for i in range(num_iter + 1):
            optimizer.zero_grad()

            net_input = net_input_saved + (noise.normal_() * reg_noise_std)
            out = net(net_input)

            if out_avg is None:
                out_avg = out.detach()
            else:
                out_avg = out_avg * exp_weight + out.detach() * (1 - exp_weight)

            total_loss = criterion(out, decrease_subimage).to(device)  # calculate the loss value of the loss function
            # print(total_loss)
            total_loss.backward()  # back propagation gradient calculation
            optimizer.step()

            if i % show_every == 0:
                msg = 'iteration times: [' + str(i) + '/' + str(num_iter) + ']'
                print(msg)
            #     out = torch.clamp(out, 0, 1)
            #     out_avg = torch.clamp(out_avg, 0, 1)
            #
            #     out_normalize = min_max_normalize(out.squeeze().detach())
            #     out_avg_normalize = min_max_normalize(out_avg.squeeze().detach())
            #     print_image([out_normalize, out_avg_normalize], title=msg, bands=[0, 1, 2])

        output.append(out.squeeze())

    # 数据重组
    output = band_recovery(group, output).cuda()
    print(output.shape)
    print_image([output.detach()])
    print(psnr(image, output))
    save_data('./data/output.mat', {'output': output.clone()})




if __name__ == '__main__':
    bands = [[181, 184, 0], [185, 161, 1], [188, 179, 2], [190, 186, 3], [183, 170, 4], [182, 187, 5],
             [174, 180, 6], [178, 173, 7], [189, 158, 8], [167, 171, 9], [177, 172, 10], [176, 163, 11],
             [165, 175, 12], [169, 168, 13], [138, 141, 14], [164, 139, 15], [140, 137, 16], [166, 162, 17],
             [22, 160, 18], [23, 135, 19], [134, 21, 20], [159, 156, 24], [26, 146, 25], [144, 145, 27],
             [150, 136, 28], [105, 131, 29], [147, 129, 30], [148, 106, 31], [143, 154, 32], [151, 149, 33],
             [142, 152, 34], [153, 133, 35], [104, 157, 36], [127, 155, 37], [132, 39, 38], [128, 130, 40],
             [45, 103, 41], [107, 50, 42], [126, 49, 43], [48, 51, 44], [53, 102, 46], [125, 121, 47],
             [108, 54, 52], [124, 119, 55], [101, 116, 56], [100, 97, 57], [123, 114, 58], [86, 122, 59],
             [99, 85, 60], [117, 120, 61], [111, 115, 62], [109, 98, 63], [65, 69, 64], [92, 87, 66], [112, 89, 67],
             [113, 96, 68], [88, 91, 70], [118, 90, 71], [95, 77, 72], [83, 76, 73], [84, 94, 74], [79, 82, 75],
             [110, 93, 78], [81, 80]]

    file_name = './data/blocks.mat'
    data_dict = read_data(file_name)
    data = []
    for key in data_dict.keys():
        data.append(data_dict[key].cuda())

    file_name = './data/denoising.mat'
    data_dict = read_data(file_name)
    origin_image = data_dict['image'].cuda()

    func(None, origin_image, data, bands)
