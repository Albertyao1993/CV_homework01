import numpy as np
import matplotlib.pyplot as plt 
# import PIL
# from PIL import Image


def display_image(img):
    """ Show an image with matplotlib:

    Args:
        Image as numpy array (H,W,3)
    """
    plt.imshow(img)
    plt.show()
 
 

def save_as_npy(path, img):
    """ Save the image array as a .npy file:

    Args:
        Image as numpy array (H,W,3)
    """
    return np.save(path,img)


def load_npy(path):
    """ Load and return the .npy file:

    Args:
        Path of the .npy file
    Returns:
        Image as numpy array (H,W,3)
    """
    return np.load(path)


def mirror_horizontal(img):
    """ Create and return a horizontally mirrored image:

    Args:
        Loaded image as numpy array (H,W,3)

    Returns:
        A horizontally mirrored numpy array (H,W,3).
    """
    Hor_img = np.copy(img)
    W = img.shape[1]

    for i in range(W):
        Hor_img[:,W - i - 1] = img[:,i]
    return Hor_img


def display_images(img1, img2):
    """ display the normal and the mirrored image in one plot:

    Args:
        Two image numpy arrays
    """
    images = [img1, img2]
    n_images = len(images)
    fig = plt.figure()
    for i in range(n_images):
        fig.add_subplot(1, n_images, i+1)
        plt.imshow(images[i])

    plt.show(block = True)