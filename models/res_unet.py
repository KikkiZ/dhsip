import torch
from torch import nn

from models.common import conv, bn, act

data_type = torch.cuda.FloatTensor


class ResUNet(nn.Module):

    def __init__(self,
                 input_channel=3,
                 output_channel=3,
                 num_channels_down=None,
                 num_channels_up=None,
                 num_channel_skip=4,
                 kernel_size_down=3,
                 kernel_size_up=3,
                 kernel_size_skip=3,
                 activate='LeakyReLU',
                 need_bias=True,
                 pad='zero',
                 upsample_mode='nearest'):
        super(ResUNet, self).__init__()

        if num_channels_up is None:
            num_channels_up = [16, 32, 64, 128, 128]
        if num_channels_down is None:
            num_channels_down = [16, 32, 64, 128, 128]

        self.encoder_1 = EncoderResBlock(input_channel=input_channel,
                                         output_channel=num_channels_down[0],
                                         skip_channel=num_channel_skip,
                                         kernel_size=kernel_size_down,
                                         kernel_size_skip=kernel_size_skip,
                                         bias=need_bias, pad=pad,
                                         activate=activate).type(data_type)
        self.encoder_2 = EncoderResBlock(input_channel=num_channels_down[0],
                                         output_channel=num_channels_down[1],
                                         skip_channel=num_channel_skip,
                                         kernel_size=kernel_size_down,
                                         kernel_size_skip=kernel_size_skip,
                                         bias=need_bias, pad=pad,
                                         activate=activate).type(data_type)
        self.encoder_3 = EncoderResBlock(input_channel=num_channels_down[1],
                                         output_channel=num_channels_down[2],
                                         skip_channel=num_channel_skip,
                                         kernel_size=kernel_size_down,
                                         kernel_size_skip=kernel_size_skip,
                                         bias=need_bias, pad=pad,
                                         activate=activate).type(data_type)
        self.encoder_4 = EncoderResBlock(input_channel=num_channels_down[2],
                                         output_channel=num_channels_down[3],
                                         skip_channel=num_channel_skip,
                                         kernel_size=kernel_size_down,
                                         kernel_size_skip=kernel_size_skip,
                                         bias=need_bias, pad=pad,
                                         activate=activate).type(data_type)
        self.encoder_5 = EncoderResBlock(input_channel=num_channels_down[3],
                                         output_channel=num_channels_down[4],
                                         skip_channel=num_channel_skip,
                                         kernel_size=kernel_size_down,
                                         kernel_size_skip=kernel_size_skip,
                                         bias=need_bias, pad=pad,
                                         activate=activate).type(data_type)

        self.decoder_5 = DecoderResBlock(input_channel=num_channels_down[4],
                                         output_channel=num_channels_up[4],
                                         kernel_size=kernel_size_up,
                                         bias=need_bias, pad=pad,
                                         activate=activate,
                                         upsample_mode=upsample_mode).type(data_type)
        self.decoder_4 = DecoderResBlock(input_channel=num_channels_up[4] + num_channel_skip,
                                         output_channel=num_channels_up[3],
                                         kernel_size=kernel_size_up,
                                         bias=need_bias, pad=pad,
                                         activate=activate,
                                         upsample_mode=upsample_mode).type(data_type)
        self.decoder_3 = DecoderResBlock(input_channel=num_channels_up[3] + num_channel_skip,
                                         output_channel=num_channels_up[2],
                                         kernel_size=kernel_size_up,
                                         bias=need_bias, pad=pad,
                                         activate=activate,
                                         upsample_mode=upsample_mode).type(data_type)
        self.decoder_2 = DecoderResBlock(input_channel=num_channels_up[2] + num_channel_skip,
                                         output_channel=num_channels_up[1],
                                         kernel_size=kernel_size_up,
                                         bias=need_bias, pad=pad,
                                         activate=activate,
                                         upsample_mode=upsample_mode).type(data_type)
        self.decoder_1 = DecoderResBlock(input_channel=num_channels_up[1] + num_channel_skip,
                                         output_channel=num_channels_up[0],
                                         kernel_size=kernel_size_up,
                                         bias=need_bias, pad=pad,
                                         activate=activate,
                                         upsample_mode=upsample_mode).type(data_type)

        self.output = conv(input_channel=num_channels_up[0],
                           output_channel=output_channel,
                           kernel_size=1,
                           bias=need_bias, pad=pad).type(data_type)

    def forward(self, data):
        out, skip_1 = self.encoder_1(data)
        out, skip_2 = self.encoder_2(out)
        out, skip_3 = self.encoder_3(out)
        out, skip_4 = self.encoder_4(out)
        out, _ = self.encoder_5(out)

        out = self.decoder_5(out)
        out = self.decoder_4(torch.cat([out, skip_4], dim=1))
        out = self.decoder_3(torch.cat([out, skip_3], dim=1))
        out = self.decoder_2(torch.cat([out, skip_2], dim=1))
        out = self.decoder_1(torch.cat([out, skip_1], dim=1))

        out = self.output(out)
        return out


class EncoderResBlock(nn.Module):

    def __init__(self,
                 input_channel,
                 output_channel,
                 skip_channel,
                 kernel_size,
                 kernel_size_skip,
                 bias,
                 pad,
                 activate):
        super(EncoderResBlock, self).__init__()

        self.reshape = conv(input_channel=input_channel,
                            output_channel=output_channel,
                            kernel_size=1, bias=bias, pad=0)

        self.block = nn.Sequential()
        self.block.add_module('0:conv', conv(input_channel=output_channel,
                                             output_channel=output_channel,
                                             kernel_size=kernel_size,
                                             bias=bias, pad=pad))
        self.block.add_module('1:bn', bn(output_channel))
        self.block.add_module('2:act', act(activate=activate))
        self.block.add_module('3:conv', conv(input_channel=output_channel,
                                             output_channel=output_channel,
                                             kernel_size=kernel_size,
                                             bias=bias, pad=pad))
        self.block.add_module('4:bn', bn(output_channel))
        self.block.add_module('5:act', act(activate=activate))

        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)

        self.skip = nn.Sequential()
        self.skip.add_module('0:conv', conv(input_channel=output_channel,
                                            output_channel=skip_channel,
                                            kernel_size=kernel_size_skip,
                                            bias=bias, pad=pad))
        self.skip.add_module('1:bn', bn(num_features=skip_channel))
        self.skip.add_module('2:act', act(activate=activate))

    def forward(self, data):
        data = self.reshape(data)
        out = self.block(data)
        out = out + data

        out = self.downsample(out)
        skip = self.skip(out)
        return out, skip


class DecoderResBlock(nn.Module):

    def __init__(self, input_channel, output_channel, kernel_size, bias, pad, activate, upsample_mode):
        super(DecoderResBlock, self).__init__()
        self.reshape = conv(input_channel=input_channel,
                            output_channel=output_channel,
                            kernel_size=1, bias=bias, pad=0)

        self.block = nn.Sequential()
        self.block.add_module('0:conv', conv(input_channel=output_channel,
                                             output_channel=output_channel,
                                             kernel_size=kernel_size,
                                             bias=bias, pad=pad))
        self.block.add_module('1:bn', bn(output_channel))
        self.block.add_module('2:act', act(activate=activate))
        self.block.add_module('3:conv', conv(input_channel=output_channel,
                                             output_channel=output_channel,
                                             kernel_size=kernel_size,
                                             bias=bias, pad=pad))
        self.block.add_module('4:bn', bn(output_channel))
        self.block.add_module('5:act', act(activate=activate))

        self.upsample = nn.Upsample(scale_factor=2, mode=upsample_mode)

    def forward(self, data):
        data = self.reshape(data)
        out = self.block(data)
        out = self.upsample(out + data)
        return out
