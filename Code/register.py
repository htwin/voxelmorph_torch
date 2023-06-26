import argparse

import numpy as np

import datagenerators
from model import cvpr2018_net, SpatialTransformer
from Funcations import save_moved_img
import torch
import nibabel as nib
def register(moving,fixed,model_path,moved_path,vm):
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    # 加载固定图像
    fixed_vol_data = datagenerators.load_volfile(fixed)
    vol_size = fixed_vol_data.shape

    fixed_image = torch.from_numpy(fixed_vol_data).to(device).float()[np.newaxis,np.newaxis,...]

    #加载移动图像
    moving_image = datagenerators.load_volfile(moving)
    moving_image = torch.from_numpy(moving_image).to(device).float()[np.newaxis,np.newaxis,...]

    # 加载模型文件
    nf_enc = [16, 32, 32, 32]
    if vm == "vm1":
        nf_dec = [32, 32, 32, 32, 8, 8]
    elif vm == "vm2":
        nf_dec = [32, 32, 32, 32, 32, 16, 16]


    model = cvpr2018_net(vol_size,nf_enc,nf_dec)
    model.to(device)
    model.load_state_dict(torch.load(model_path,map_location=lambda storage, loc: storage))

    moved,flow = model(moving_image,fixed_image)

    # 加载的原始对象(保存配准图像时，需要使用它的空间信息，确保配准图像空间信息和固定图像一致)
    fixed_vol_file = nib.load(fixed)
    save_moved_img(moved,fixed_vol_file,moved_path)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--moving',type=str,help='移动图像',default='../Data/moving_aligned_norm.nii.gz')
    parser.add_argument('--fixed',type=str,help='固定图像',default='../Data/fixed_aligned_norm.nii.gz')
    parser.add_argument('--model_path',type=str,help='模型文件',default='../Models/cvpr2018_vm2_l2_pytorch.ckpt')
    parser.add_argument('--moved_path',type=str,help='配准图像的输出',default='../Data/moved.nii.gz')
    parser.add_argument('--vm',type=str,help='vm1 or vm2',default='vm2')
    register(**vars(parser.parse_args()))