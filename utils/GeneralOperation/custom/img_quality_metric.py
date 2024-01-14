import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim


def psnr_single(im1: np.ndarray, im2: np.ndarray) -> float:
    assert im1.shape == im2.shape, "im1 and im2 should have the same shape"
    assert im1.dtype == im2.dtype, "im1 and im2 should have the same dtype"
    R = 1 if im1.dtype == int else 255
    psnr_score = cv2.PSNR(im1, im2, R=R)
    return psnr_score


def ssim_single(im1: np.ndarray, im2: np.ndarray) -> float:
    assert im1.shape == im2.shape, "im1 and im2 should have the same shape"
    assert im1.dtype == im2.dtype, "im1 and im2 should have the same dtype"
    data_range = 255 if im1.dtype == int else 1
    # assert im2.dtype == float
    if im1.ndim == 3 and im2.ndim == 3:  # RGB images
        ssim_score = ssim(
            im1, im2, multichannel=True, data_range=data_range, gaussian_weights=True
        )
    # gray images, recommended for higher ssim scores.
    elif im1.ndim == 2 and im2.ndim == 2:
        ssim_score = ssim(
            im1, im2, multichannel=False, data_range=data_range, gaussian_weights=True
        )
    return ssim_score


def psnr_batch(imb1: np.ndarray, imb2: np.ndarray):
    """PSNR scores for an image batch

    Args:
        imb1 (np.ndarray):  N*W*H*D
        imb2 (np.ndarray):  N*W*H*D

    Returns:
        mean psnr score, psnr score list
    """
    psnr_scores = np.array([psnr_single(i[0], i[1]) for i in zip(imb1, imb2)])
    return np.mean(psnr_scores), psnr_scores


def ssim_batch(imb1: np.ndarray, imb2: np.ndarray):
    """SSIM scores for an image batch

    Args:
        imb1 (np.ndarray):  N*W*H*D
        imb2 (np.ndarray):  N*W*H*D

    Returns:
        mean ssim score, ssim score list
    """
    ssim_scores = np.array([ssim_single(i[0], i[1]) for i in zip(imb1, imb2)])
    return np.mean(ssim_scores), ssim_scores


def print_similarity_results(imb1: np.ndarray, imb2: np.ndarray):
    mean_psnr, _ = psnr_batch(imb1, imb2)
    mean_ssim, _ = ssim_batch(imb1, imb2)
    print(
        f"Similarity results => mean PSNR: {mean_psnr:.4f};  mean SSIM: {mean_ssim:.4f}"
    )


def norm_data(x:np.ndarray) -> np.ndarray:
    """
    @Desc: To normalize an image array;
    @Args:
        x: Image array;
    @Return:
        norm_x: Normalised image array;
    """
    mean_data=np.mean(x)
    std_data=np.std(x, ddof=1)
    if std_data == 0:
        std_data = 1e-6
    norm_x = (x-mean_data)/(std_data)
    return norm_x


def normalized_cross_correlation(x_0:np.ndarray, x_1:np.ndarray) -> float:
    """
    @Desc: To compute normalized cross-correlation coefficient between two image array;
    @Args:
        x_0: Image array 0;
        x_1: Image array 1;
    @Return:
        ncc: Normalized cross correlation;
    """
    ncc = (1.0/(x_0.size-1)) * np.sum(norm_data(x_0)*norm_data(x_1))
    ncc = np.abs(ncc)
    return ncc
