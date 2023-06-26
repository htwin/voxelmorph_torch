"""
*Preliminary* pytorch implementation.

VoxelMorph training.
"""


# python imports
import os
import glob
import random
import warnings
from argparse import ArgumentParser

# external imports
import numpy as np
import torch
from torch.optim import Adam

# internal imports
from model import cvpr2018_net
import datagenerators
import losses


def train(gpu,
          data_dir,
          atlas_file,
          lr,
          n_iter,
          data_loss,
          model,
          reg_param, 
          batch_size,
          n_save_iter,
          model_dir):
    """
    model training function
    :param gpu: integer specifying the gpu to use
    :param data_dir: folder with npz files for each subject.
    :param atlas_file: atlas filename. So far we support npz file with a 'vol' variable
    :param lr: learning rate
    :param n_iter: number of training iterations
    :param data_loss: data_loss: 'mse' or 'ncc
    :param model: either vm1 or vm2 (based on CVPR 2018 paper)
    :param reg_param: the smoothness/reconstruction tradeoff parameter (lambda in CVPR paper)
    :param batch_size: Optional, default of 1. can be larger, depends on GPU memory and volume size
    :param n_save_iter: Optional, default of 500. Determines how many epochs before saving model version.
    :param model_dir: the model directory to save to
    """

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = "cuda"

    # Produce the loaded atlas with dims.:160x192x224.
    atlas_vol = datagenerators.load_volfile(atlas_file)
    vol_size = atlas_vol.shape

    # Get all the names of the training data
    # 训练文件使用1到255的文件 第0个文件已经作为固定图像（参数中的atlas_file）
    train_vol_names = sorted(glob.glob(data_dir + '/OASIS_OAS1_*_MR1/aligned_norm.nii.gz'))[1:255]
    # random.shuffle(train_vol_names)

    # Prepare the vm1 or vm2 model and send to device
    nf_enc = [16, 32, 32, 32]
    if model == "vm1":
        nf_dec = [32, 32, 32, 32, 8, 8]
    elif model == "vm2":
        nf_dec = [32, 32, 32, 32, 32, 16, 16]
    else:
        raise ValueError("Not yet implemented!")

    model = cvpr2018_net(vol_size, nf_enc, nf_dec)
    model.to(device)

    # Set optimizer and losses
    opt = Adam(model.parameters(), lr=lr)

    sim_loss_fn = losses.ncc_loss if data_loss == "ncc" else losses.mse_loss
    grad_loss_fn = losses.gradient_loss

    # data generator
    train_example_gen = datagenerators.example_gen(train_vol_names, batch_size)

    # set up atlas tensor
    input_fixed  = torch.from_numpy(atlas_vol).to(device).float()[np.newaxis, np.newaxis, ...,]

    # Training loop.
    for i in range(n_iter):

        # Save model checkpoint
        if i % n_save_iter == 0:
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            save_file_name = os.path.join(model_dir, '%d.ckpt' % i)
            torch.save(model.state_dict(), save_file_name)

        # Generate the moving images and convert them to tensors.
        moving_image = next(train_example_gen)[0]
        input_moving = torch.from_numpy(moving_image).to(device).float()
        input_moving = input_moving.permute(0, 4, 1, 2, 3)
        # Run the data through the model to produce warp and flow field
        warp, flow = model(input_moving, input_fixed)

        # Calculate loss
        recon_loss = sim_loss_fn(warp, input_fixed) 
        grad_loss = grad_loss_fn(flow)
        loss = recon_loss + reg_param * grad_loss

        print("%d,%f,%f,%f" % (i, loss.item(), recon_loss.item(), grad_loss.item()), flush=True)

        # Backwards and optimize
        opt.zero_grad()
        loss.backward()
        opt.step()


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)

    parser = ArgumentParser()

    parser.add_argument("--gpu",
                        type=str,
                        default='0',
                        help="gpu id")

    parser.add_argument("--data_dir",
                        type=str,
                        default='D:/code/cv/datasets/original/neurite-oasis.v1.0',
                        help="data folder with training vols")

    parser.add_argument("--atlas_file",
                        type=str,
                        dest="atlas_file",
                        default='D:/code/cv/datasets/original/neurite-oasis.v1.0/OASIS_OAS1_0001_MR1/aligned_norm.nii.gz',
                        help="gpu id number")

    parser.add_argument("--lr",
                        type=float,
                        dest="lr",
                        default=1e-4,
                        help="learning rate")

    parser.add_argument("--n_iter",
                        type=int,
                        dest="n_iter",
                        default=150000,
                        help="number of iterations")

    parser.add_argument("--data_loss",
                        type=str,
                        dest="data_loss",
                        default='ncc',
                        help="data_loss: mse of ncc")

    parser.add_argument("--model",
                        type=str,
                        dest="model",
                        choices=['vm1', 'vm2'],
                        default='vm2',
                        help="voxelmorph 1 or 2")

    parser.add_argument("--lambda", 
                        type=float,
                        dest="reg_param", 
                        default=0.01,  # recommend 1.0 for ncc, 0.01 for mse
                        help="regularization parameter")

    parser.add_argument("--batch_size", 
                        type=int,
                        dest="batch_size", 
                        default=1,
                        help="batch_size")

    parser.add_argument("--n_save_iter", 
                        type=int,
                        dest="n_save_iter", 
                        default=500,
                        help="frequency of model saves")

    parser.add_argument("--model_dir", 
                        type=str,
                        dest="model_dir", 
                        default='../models/',
                        help="models folder")


    train(**vars(parser.parse_args()))

