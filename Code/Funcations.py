import numpy as np
import nibabel as nib
# 计算dice的函数
def dice(vol1, vol2, labels=None, nargout=1):
    if labels is None:
        labels = np.unique(np.concatenate((vol1, vol2)))
        labels = np.delete(labels, np.where(labels == 0))  # remove background
    dicem = np.zeros(len(labels))
    for idx, lab in enumerate(labels):
        vol1l = vol1 == lab
        vol2l = vol2 == lab
        top = 2 * np.sum(np.logical_and(vol1l, vol2l))
        bottom = np.sum(vol1l) + np.sum(vol2l)
        bottom = np.maximum(bottom, np.finfo(float).eps)  # add epsilon.
        dicem[idx] = top / bottom

    if nargout == 1:
        return dicem
    else:
        return (dicem, labels)




# 保存配准图像
def save_moved_img(I_img,fixd_vol_file, savename):
    """
    I_img：需要保存的图像
    fixd_vol_file：固定图像 保存配准图像时需要使用固定图像的空间信息
    savename：保存文件名 也就是文件全路径
    """
    I_img = I_img[0, 0, ...].cpu().detach().numpy()
    # 使用固定图像的affine矩阵，描述了图像在物理空间中的位置和方向
    affine = fixd_vol_file.affine
    new_img = nib.nifti1.Nifti1Image(I_img, affine, header=None)
    nib.save(new_img, savename)


# 保存nii 图像
def save_img_nii(I_img, savename):
    I_img = I_img[0, 0, ...].cpu().detach().numpy()
    affine = np.diag([1, 1, 1, 1])
    new_img = nib.nifti1.Nifti1Image(I_img, affine, header=None)
    nib.save(new_img, savename)

