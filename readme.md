# Voxelmorch-PyTorch 快速上手使用版本

## 目录结构

- Code 存放源代码的文件
  - datagenerators.py 用于加载图像文件的数据
  - Funcations 一些工具函数实现 比如计算dice系数的函数等
  - losses 损失函数的实现
  - model 配准网络的实现 包括U-Net和STN
  - register.py 配准代码的实现，输入移动图像、固定图像、模型文件， 输出配准图像。
  - test.py 测试代码
  - train.py 训练代码
- Data 提供一对图像文件，可快速运行配准代码
  - fixed_aligned_norm.nii.gz  固定图像
  - fixed_aligned_seg35.nii.gz 包含有分割标签的固定图像
  - moving_aligned_norm.nii.gz 移动图像
  - moved.nii.gz 运行配准代码后 保存的配准后的图像
- Models  存放训练好的模型文件
  - cvpr2018_vm2_l2_pytorch.ckpt 官网提供的训练好的模型文件
  -  17500.ckpt  我训练17500轮的模型 效果比官方要差一点
- Testfiles 存放了三组用于测试的文件
- Datasets 数据集文件



## 配准

### 自定义参数运行

- --moving  移动图像文件地址
- --fixed 固定图像地址
- --model_path 模型文件地址
- --moved_path 配准图像保存地址

```python
# 进入到Code文件夹(cd Code) 运行配准文件register.py 后接参数
python ./register.py --moving ../Data/moving_aligned_norm.nii.gz --fixed ../D
ata/fixed_aligned_norm.nii.gz --model_path ../Models/cvpr2018_vm2_l2_pytorch.ckpt --moved_path ../Data/moved.nii.gz
```

### 直接运行

直接运行register.py，默认会使用Data目录中的数据进行配准

```python
# 进入到Code文件夹(cd Code) 直接运行配准文件
python ./register.py
```



## 测试

### 自定义参数运行

```python
python test.py
# 后接下面的参数，基本上都有默认值，根据自己实际情况设置参数
```

- --gpu 默认为0 使用gpu 0
- --data_dir 数据集路径 默认是Datasets文件夹中的neurite-oasis.v1.0
- --atlas_file  atlas文件， 默认为Data下的fixed...文件
- --atlas_label atlas文件标签文件 ，默认为Data下的......seg35.nii.gz
- --model 模型版本 vm1或者vm2 默认 vm2
- --init_model_file 默认文件路径 默认为Models下的官方模型
- --test_dir 测试文件夹 默认为Testfiles文件夹
- --label_dir 标签文件夹 默认为Testfiles 文件夹

### 直接运行

同上，进入Code文件夹，直接运行test.py,默认会使用Testfiles文件夹中的三个测试文件进行测试运行。





## 训练

### 自定义参数运行

```python
python tain.py 
# 后接下面的参数，基本上都有默认值，根据自己实际情况设置参数
```

- --gpu 默认为0 使用gpu 0
- --data_dir 数据集路径 默认是Datasets文件夹中的neurite-oasis.v1.0
- --atlas_file  atlas文件， 训练使用的固定文件 默认为Data下的fixed...文件
- --lr 学习速率 默认1e-4
- --n_iter 迭代次数 默认15000
- --data_loss 损失函数 ncc或者mse 默认ncc
- --model 模型版本 vm1或者vm2 默认 vm2
- --lambda 正则化参数 ncc默认为1.0 mse默认为0.01
-  --batch_size 批处理样本数量 默认为1
- --n_save_iter 默认保存频率 默认500 (训练500轮保存一次)
- --model_dir 模型文件保存的路径

### 直接运行

同上，数据集整理好之后，直接运行train.py 文件，数据集默认会加载Datasets中的数据，如果数据集存放位置不同，可自己修改代码。