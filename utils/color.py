import numpy as np

# Functions below translated from MATLAB to Python from
# https://github.com/STAC-USC/RA-GFT

Q_RGBtoYUV = np.array([
        [0.29899999,    -0.1687,         0.5],
        [0.587,         -0.3313,        -0.4187],
        [0.114,          0.5,           -0.0813],
        [0,              0.50196078,     0.50196078]
])

M_YUVtoRGB = np.array([
        [1,             1,           1],
        [0,            -0.34414,     1.772],
        [1.402,        -0.71414,     0],
        [-0.703749019,   0.53121505, -0.88947451]
])


def rgb_to_yuv(rgb, rounding=False):

    # Limit values between 0 and 1 (instead of 0 and 255)
    # Append column filled with ones
    rgb1 = np.concatenate(
        (rgb/255, np.ones((rgb.shape[0], 1))), axis=1)

    # Conversion
    yuv = np.dot(rgb1, Q_RGBtoYUV)

    # Limit values between 0 and 255
    yuv = 255*np.clip(yuv, 0, 1)

    # Round to integer values
    if rounding:
        yuv = yuv.round().astype(np.uint8)

    return yuv


def yuv_to_rgb(yuv, rounding=True):

    # Limit values between 0 and 1 (instead of 0 and 255)
    # Append column filled with ones
    yuv1 = np.concatenate(
        (yuv/255, np.ones((yuv.shape[0], 1))), axis=1)

    # Conversion
    rgb = np.dot(yuv1, M_YUVtoRGB)

    # Limit values between 0 and 255
    rgb = (255*np.clip(rgb, 0, 1))

    # Round to integer values
    if rounding:
        rgb = rgb.round().astype(np.uint8)

    return rgb

def rgb_to_lab(rgb):
    from skimage import color as skcolor

    return skcolor.rgb2lab(rgb)


def lab_to_rgb(lab):
    from skimage import color as skcolor

    return skcolor.lab2rgb(lab)


def YPSNR(A, Aq, N):
    return -10*np.log10(
        np.linalg.norm(A[:, 0] - Aq[:, 0])**2/(N*255**2)
    )
