from torch import nn

from models.common import conv, act, bn


class ResNet(nn.Module):

    def __init__(self,
                 input_channel=3,
                 output_channel=3,
                 num_channels_in=None,
                 num_channels_out=None,
                 kernel_size=3,
                 activate='LeakyReLU',
                 need_bias=True,
                 pad='zero'):

        super(ResNet, self).__init__()

        if num_channels_in is None:
            num_channels_in = [16, 32, 64, 128, 128]
        if num_channels_out is None:
            num_channels_out = [16, 32, 64, 128, 128]

        self.net = nn.Sequential()

        for layer in range(0, len(num_channels_in)):
            block = ResBlock(num_channels_in[layer],
                             num_channels_out[layer],
                             kernel_size=kernel_size,
                             bias=need_bias,
                             pad=pad,
                             activate=activate)

            self.net.add_module('layer:' + str(layer), block)

        self.input_layer = conv(input_channel, num_channels_in[0], kernel_size=kernel_size, pad=pad)
        self.output_layer = conv(num_channels_out[len(num_channels_out) - 1],
                                 output_channel, kernel_size=kernel_size, pad=pad)

    def forward(self, data):
        data = self.input_layer(data)
        data = self.net(data)
        data = self.output_layer(data)

        return data


class ResBlock(nn.Module):

    def __init__(self, input_channel, output_channel, kernel_size, bias, pad, activate):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential()
        self.block.add_module('0:conv', conv(input_channel=input_channel,
                                             output_channel=output_channel,
                                             kernel_size=kernel_size,
                                             bias=bias,
                                             pad=pad))
        self.block.add_module('1:bn', bn(output_channel))
        self.block.add_module('2:act', act(activate=activate))
        self.block.add_module('3:conv', conv(input_channel=output_channel,
                                             output_channel=output_channel,
                                             kernel_size=kernel_size,
                                             bias=bias,
                                             pad=pad))
        self.block.add_module('4:bn', bn(output_channel))

        self.activate = act(activate=activate)

    def forward(self, data):
        out = self.block(data)
        out = self.activate(data + out)
        return out
