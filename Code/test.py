"""
*Preliminary* pytorch implementation.

VoxelMorph testing
"""


# python imports
import os
import glob
from argparse import ArgumentParser

import numpy as np
import torch
from model import cvpr2018_net, SpatialTransformer
import datagenerators
from Funcations import dice

#  需要计算的标签类别
# 这儿使用的 fixed_aligned_seg35.nii.gz 来计算dice
good_labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
               26, 27, 28, 29, 30, 31, 32, 33, 34, 35]


def test(gpu, 
         atlas_file,
         atlas_label,
         test_dir,
         label_dir,
         model, 
         init_model_file):

    """
    参数
        gpu: 指定使用的gpu 默认为 0
        atlas_file: 固定图像文件
        atlas_label： 固定图像的标签 文件
        test_dir： 测试文件路径
        label_dir： 测试文件的标签路径
        model：vm1或者vm2 默认 vm2
        init_model_file：加载的模型文件
    """
    device = 'cpu'
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        device = "cuda"

    # 加载固定图像
    atlas_vol = datagenerators.load_volfile(atlas_file)
    input_fixed = torch.from_numpy(atlas_vol).to(device).float()[np.newaxis, np.newaxis, ...]
    # 固定图像对应的label
    fixed_label = datagenerators.load_volfile(atlas_label)
    vol_size = atlas_vol.shape

    # 测试文件
    test_file_lst = sorted(glob.glob(test_dir + '/OASIS_OAS1_*_MR1/aligned_norm.nii.gz'))


    # Prepare the vm1 or vm2 model and send to device
    nf_enc = [16, 32, 32, 32]
    if model == "vm1":
        nf_dec = [32, 32, 32, 32, 8, 8]
    elif model == "vm2":
        nf_dec = [32, 32, 32, 32, 32, 16, 16]

    # 加载模型
    model = cvpr2018_net(vol_size, nf_enc, nf_dec)
    model.to(device)
    model.load_state_dict(torch.load(init_model_file, map_location=lambda storage, loc: storage))


    # Use this to warp segments
    trf = SpatialTransformer(atlas_vol.shape, mode='nearest')
    trf.to(device)

    for file in test_file_lst:
        # moving图像
        input_moving = datagenerators.load_volfile(file)
        input_moving = torch.from_numpy(input_moving).to(device).float()[np.newaxis, np.newaxis, ...]

        # 得到配准后的图像和形变场
        warp, flow = model(input_moving, input_fixed)

        # 读入moving图像对应的label
        filename_pre = os.path.split(file)[0].split(os.path.sep)[-1]
        label_file = glob.glob(os.path.join(label_dir, filename_pre, "aligned_seg35.nii.gz"))[0]
        moving_seg = datagenerators.load_volfile(label_file)
        moving_seg = torch.from_numpy(moving_seg).to(device).float()[np.newaxis, np.newaxis, ...]
        warp_seg = trf(moving_seg, flow).detach().cpu().numpy()
        # 计算dice
        vals, labels = dice(warp_seg, fixed_label, labels=good_labels, nargout=2)
        #dice_vals[:, k] = vals
        #print(np.mean(dice_vals[:, k]))
        print("moving_image:"+file)
        print("fixed_image:"+atlas_file)
        print("dice:",np.mean(vals))

        #return

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--gpu",
                        type=str,
                        default='0',
                        help="gpu id")

    parser.add_argument("--atlas_file",
                        type=str,
                        dest="atlas_file",
                        default='../Data/fixed_aligned_norm.nii.gz',
                        help="gpu id number")

    parser.add_argument("--atlas_label",
                        type=str,
                        dest="atlas_label",
                        default='../Data/fixed_aligned_seg35.nii.gz',
                        help="gpu id number")

    parser.add_argument("--model",
                        type=str,
                        dest="model",
                        choices=['vm1', 'vm2'],
                        default='vm2',
                        help="voxelmorph 1 or 2")

    parser.add_argument("--init_model_file", 
                        type=str,
                        default="../Models/cvpr2018_vm2_l2_pytorch.ckpt",
                        dest="init_model_file", 
                        help="model weight file")

    parser.add_argument("--test_dir",
                        type=str,
                        dest="test_dir",
                        default="../Testfiles",
                        help="test data directory")
    parser.add_argument("--label_dir",
                        type=str,
                        dest="label_dir",
                        default="../Testfiles",
                        help="label data directory")

    test(**vars(parser.parse_args()))

