# dhsip

## 快速开始

### 环境要求

* Python >= 3.9
* PyTorch >= 2.1
* NumPy
* scipy
* scikit-image

上述为该库必须使用的包, 对于兼容性问题, 该库没有进行测试. 对于老版本的Python和PyTorch, 我们不推荐使用, 但或许可以兼容. 同时, 该库的代码中大量使用了CUDA和CUDNN来提升模型的运行速率, 假如您的计算机没有GPU加速训练时, 需要修改代码以便模型正常运行.

### 数据准备

* 测试主要使用了 `191 Band Hyperspectral Image: HYDICE image of Washington DC Mall`，你可以在 https://engineering.purdue.edu/~biehl/MultiSpec/hyperspectral.html 获得获得公开的数据集.
* 在 `/data/data_process.py` 中准备了一些数据处理的函数，你可以自由的调用这些函数，以生成适合模型的数据. 这些函数已经写了详细的文档注释，可以通过这些注释了解使用方法.

> 注意: 该库使用的数据持久化的方式是PyTorch由提供的, 我们推荐您使用 `.pth` 存储和读取数据

### 运行代码

在准备好合适的数据之后, 就可以运行模型. `bootstrap.py` 是模型启动的入口, 你在启动时需要附带一些参数以便模型正常的初始化, 下面是代码能够接受的参数以及类型:

* `--net`: 神经网络的类型, 你可以在 `unet`|`res`|`res-unet` 中选择;
* `--mode`: 模型运行的模式, 你可以在 `base`|`red`|`band` 中选择;
* `--num_iter`: 模型迭代次数, 任意一个整数(我们建议的取值范围是 1000-10000);
* `--reg_noise_std`: 模型每次迭代输入噪声张量的扰动系数, 任意一个浮点数(我们建议使用默认值, 取值范围在 0.03左右);
* `--show_every`: 模型展示当前轮次模型运行结果的频率, 任意一个整数(我们建议的取值是 >=100), 如果不需要展示当前轮次的运行结果可以设置一个较大的值;
* `--exp_weight`: 模型输出平滑系数, 任意一个浮点数(我们建议使用默认值, 取值范围在0.99左右);
* `--lr`: 模型的学习率;
* `--skip_channel`: 使用unet时跳跃连接的通道数, 任意一个整数;
* `--up_channel`: unet模型解码器的通道数, 五个正整数;
* `--down_channel`: unet模型编码器的通道数, 五个正整数;
* `--upsample_mode`: unet模型上采样的模式;
* `--downsample_mode`: unet模型下采样的模式;
* `--group_size`: 波段分组模式中每组的波段数量;

你也可以使用下面的指令快速开始:

```cmd
python bootstrap.py --net unet --mode base --num_iter 1800 --show_every 200 --skip_channel 128 --up_channel  128 128 128 128 128 --down_channel  128 128 128 128 128
```

## 致谢

该库借鉴和使用了很多研究者的代码或研究，感谢他们的贡献. 以下排名不分先后：

* [DmitryUlyanov/deep-image-prior: Image restoration with neural networks but without learning (GitHub.com)](https://github.com/DmitryUlyanov/deep-image-prior)
* [acecreamu/deep-hs-prior: Single Hyperspectral Image Denoising, Inpainting, Super-Resolution (GitHub.com)](https://github.com/acecreamu/deep-hs-prior)
* [GaryMataev/DeepRED: DeepRED: Deep Image Prior Powered by RED (GitHub.com)](https://github.com/GaryMataev/DeepRED)

