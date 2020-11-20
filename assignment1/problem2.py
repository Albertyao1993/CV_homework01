import numpy as np
from scipy.ndimage import convolve


def loaddata(path):
    """ Load bayerdata from file

    Args:
        Path of the .npy file
    Returns:
        Bayer data as numpy array (H,W)
    """
    return np.load(path)

def separatechannels(bayerdata):
    """ Separate bayer data into RGB channels so that
    each color channel retains only the respective
    values given by the bayer pattern and missing values
    are filled with zero

    Args:
        Numpy array containing bayer data (H,W)
    Returns:
        red, green, and blue channel as numpy array (H,W)
    """
    H, W = bayerdata.shape
    r_img = np.zeros((H, W))
    g_img = np.zeros((H, W))
    b_img = np.zeros((H, W))
    #base on RGB forme from exercise sheet
    r_img[0:H:2, 1:W:2] = bayerdata[0:H:2, 1:W:2]
    b_img[1:H:2, 0:W:2] = bayerdata[1:H:2, 0:W:2]
    g_img[0:H:2, 0:W:2] = bayerdata[0:H:2, 0:W:2]
    g_img[1:H:2, 1:W:2] = bayerdata[1:H:2, 1:W:2]

    #base on RGB form from lecture02 page 37
    # r_img[1:H:2, 1:W:2] = bayerdata[1:H:2, 1:W:2]
    # b_img[0:H:2, 0:W:2] = bayerdata[0:H:2, 0:W:2]
    # g_img[0:H:2, 1:W:2] = bayerdata[0:H:2, 1:W:2]
    # g_img[1:H:2, 0:W:2] = bayerdata[1:H:2, 0:W:2]

    return r_img, g_img, b_img


def assembleimage(r, g, b):
    """ Assemble separate channels into image

    Args:
        red, green, blue color channels as numpy array (H,W)
    Returns:
        Image as numpy array (H,W,3)
    """

    return np.dstack((r,g,b))


def interpolate(r, g, b):
    """ Interpolate missing values in the bayer pattern
    by using bilinear interpolation

    Args:
        red, green, blue color channels as numpy array (H,W)
    Returns:
        Interpolated image as numpy array (H,W,3)
    """
    # G = np.array([[1,2,1],[2,4,2],[1,2,1]])
    # Gauss_kernel = G/np.sum(G)
    kernel = 1/4*np.array([
        [1,0,1],
        [0,0,0],
        [1,0,1]
    ])
    # feels better with nearest 
    r = convolve(r, kernel, mode='nearest')
    g = convolve(g, kernel, mode='nearest') 
    b = convolve(b, kernel, mode='nearest')
    # r = convolve(r, Gauss_kernel, mode='constant')
    # g = convolve(g, Gauss_kernel, mode='constant') 
    # b = convolve(b, Gauss_kernel, mode='constant')
    return np.dstack((r,g,b))


