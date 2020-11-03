import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.signal import convolve2d
from scipy import signal

def load_data(path):
    '''
    Load data from folder data, face images are in the folder facial_images, face features are in the folder facial_features.
    

    Args:
        path: path of folder data

    Returns:
        imgs: list of face images as numpy arrays 
        feats: list of facial features as numpy arrays 
    '''
    imgs = []
    feats = []

    for img in os.listdir(path+'/facial_images'):
        # I = Image.open(item)
        # imgs.append(np.array(I))
        imgs.append(plt.imread(path+'/facial_images'+'/'+img))
        # imgs.append(mpimg.imread(item))
    
    for feat in os.listdir(path+'/facial_features'):
        # feats.append(mpimg.imread(item))
        # I = Image.open(item)
        # feats.append(np.array(I))
        feats.append(plt.imread(path+'/facial_features'+'/'+feat))
    # print(imgs)
    # print(feats)
    return imgs, feats

def gaussian_kernel(fsize, sigma):
    '''
    Define a Gaussian kernel

    Args:
        fsize: kernel size
        sigma: sigma of Gaussian kernel

    Returns:
        The Gaussian kernel
    '''

    #
    # TODO
    #

    gaussian_kernel_1d = signal.gaussian(fsize,sigma).reshape(fsize,1)
    gaussian_kernel_2d = np.outer(gaussian_kernel_1d,gaussian_kernel_1d)
    return gaussian_kernel_2d
 


def downsample_x2(x, factor=2):
    '''
    Downsampling an image by a factor of 2

    Args:
        x: image as numpy array (H * W)

    Returns:
        downsampled image as numpy array (H/2 * W/2)
    '''

    #
    # TODO
    #


    # downsample = np.empty(None, None)
    downsample = x[::factor,::factor]

    return downsample


def gaussian_pyramid(img, nlevels, fsize, sigma):
    '''
    A Gaussian pyramid is constructed by combining a Gaussian kernel and downsampling.
    Tips: use scipy.signal.convolve2d for filtering image.

    Args:
        img: face image as numpy array (H * W)
        nlevels: number of levels of Gaussian pyramid, in this assignment we will use 3 levels
        fsize: Gaussian kernel size, in this assignment we will define 5
        sigma: sigma of Gaussian kernel, in this assignment we will define 1.4

    Returns:
        GP: list of Gaussian downsampled images, it should be 3 * H * W
    '''
    GP = []

    #
    # TODO
    #
    img_temp = img.copy()
    gaussian = gaussian_kernel(fsize,sigma)
    GP.append(img)
    for i in range(nlevels-1):
        img_temp = convolve2d(img_temp,gaussian, mode='same', boundary='fill',fillvalue=0)
        GP.append(downsample_x2(img_temp))
        
    return GP

def template_distance(v1, v2):
    '''
    Calculates the distance between the two vectors to find a match.
    Browse the course slides for distance measurement methods to implement this function.
    Tips: 
        - Before doing this, let's take a look at the multiple choice questions that follow. 
        - You may need to implement these distance measurement methods to compare which is better.

    Args:
        v1: vector 1
        v2: vector 2

    Returns:
        Distance
    '''
    distance  = np.sum((v1 - v2) ** 2)

    #
    # TODO
    #

    return distance 


def sliding_window(img, feat, step=1):
    ''' 
    A sliding window for matching features to windows with SSDs. When a match is found it returns to its location.
    
    Args:
        img: face image as numpy array (H * W)
        feat: facial feature as numpy array (H * W)
        step: stride size to move the window, default is 1
    Returns:
        min_score: distance between feat and window
    '''

    min_score = None

    #
    # TODO
    #
    img_h,img_w = img.shape
    feat_h,feat_w = feat.shape
    for i in range(0,img_h - feat_h+1,step):
        for j in range(0,img_w - feat_w+1, step):
            score = [template_distance(feat, img[i:i+feat_h,j:j+feat_w])]

    min_score = min(score)

    return min_score


class Distance(object):

    # choice of the method
    METHODS = {1: 'Dot Product', 2: 'SSD Matching'}

    # choice of reasoning
    REASONING = {
        1: 'it is more computationally efficient',
        2: 'it is less sensitive to changes in brightness.',
        3: 'it is more robust to additive Gaussian noise',
        4: 'it can be implemented with convolution',
        5: 'All of the above are correct.'
    }

    def answer(self):
        '''Provide your answer in the return value.
        This function returns one tuple:
            - the first integer is the choice of the method you will use in your implementation of distance.
            - the following integers provide the reasoning for your choice.
        Note that you have to implement your choice in function template_distance

        For example (made up):
            (1, 1) means
            'I will use Dot Product because it is more computationally efficient.'
        '''

        return (2, 5)  # TODO


def find_matching_with_scale(imgs, feats):
    ''' 
    Find face images and facial features that match the scales 
    
    Args:
        imgs: list of face images as numpy arrays
        feats: list of facial features as numpy arrays 
    Returns:
        match: all the found face images and facial features that match the scales: N * (score, g_im, feat)
        score: minimum score between face image and facial feature
        g_im: face image with corresponding scale
        feat: facial feature
    '''
    match = []
    # (score, g_im, feat) = (None, None, None)

    #
    # TODO
    #
 
    for feat in feats:
        temp_dist = []
        list_dist = []
        for img in imgs:
            GP = gaussian_pyramid(img, nlevels=3, fsize=5, sigma=1.4)
            print(g for g in GP)
            for g in GP:
                distance = sliding_window(g,feat,step=1)
                temp_dist.append((distance,g,feat))
                list_dist.append(distance)
        index = np.argmin(list_dist)                
        match.append(temp_dist[index])

    print(match)


    return match