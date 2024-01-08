import torch

from models.res_unet import ResUNet
from models.resnet import ResNet
from models.unet import UNet


def noise_generator(method, shape: list) -> torch.Tensor:
    """ 生成一个指定大小的噪声张量

    :param method: 需要生成张量的维度
    :param shape: 张量的形状
    :return: 返回指定大小的噪声张量
    """
    if method == '2D':
        shape.insert(0, 1)
    elif method == '3D':
        shape.insert(0, 1)
        shape.insert(0, 1)
    else:
        assert False

    noise_tensor = torch.zeros(shape)
    noise_tensor.uniform_()
    noise_tensor *= 0.1

    return noise_tensor


def model_build(args, image) -> torch.nn.Module:
    """ 构建网络模型
    根据参数构建指定的网络模型

    :param args: 程序输入参数
    :param image: 模型输入数据
    :return: 返回网络模型
    """

    if args.net == 'unet':
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

    elif args.net == 'res-unet':
        net = ResUNet(image.shape[0],
                      image.shape[0],
                      num_channels_up=args.up_channel,
                      num_channels_down=args.down_channel,
                      num_channel_skip=image.shape[0],
                      kernel_size_up=3,
                      kernel_size_down=3,
                      kernel_size_skip=3,
                      upsample_mode=args.upsample_mode,
                      need_bias=True,
                      pad='reflection',
                      activate='LeakyReLU')

    else:
        raise ValueError('The input parameter --net is incorrect, you need to choose between unet, res and res-unet')

    return net
