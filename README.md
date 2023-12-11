# dhsip

## 快速开始

### 环境要求

* Python >= 3.9
* PyTorch >= 2.1
* NumPy
* scipy
* scikit-image

上述为该库必须使用的包, 对于兼容性问题, 该库没有进行测试. 对于老版本的Python和PyTorch, 我们不推荐使用, 但或许可以兼容. 同时, 该库的代码中大量使用了CUDA和CUDNN来提升模型的运行速率, 假如您的计算机没有GPU加速训练时, 需要修改代码以便模型正常运行.