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

* 测试主要使用了`191 Band Hyperspectral Image: HYDICE image of Washington DC Mall`，你可以在 https://engineering.purdue.edu/~biehl/MultiSpec/hyperspectral.html 获得获得公开的数据集.
* 在`/data/data_process.py`中准备了一些数据处理的函数，你可以自由的调用这些函数，以生成适合模型的数据. 这些函数已经写了详细的文档注释，可以通过这些注释了解使用方法.

## 致谢

该库借鉴和使用了很多研究者的代码或研究，感谢他们的贡献. 以下排名不分先后：

* [DmitryUlyanov/deep-image-prior: Image restoration with neural networks but without learning. (github.com)](https://github.com/DmitryUlyanov/deep-image-prior)
* [acecreamu/deep-hs-prior: Single Hyperspectral Image Denoising, Inpainting, Super-Resolution (github.com)](https://github.com/acecreamu/deep-hs-prior)
* [GaryMataev/DeepRED: DeepRED: Deep Image Prior Powered by RED (github.com)](https://github.com/GaryMataev/DeepRED)

