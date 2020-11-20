import math
import numpy as np
from scipy import ndimage


def gauss2d(sigma, fsize):
  """
  Args:
    sigma: width of the Gaussian filter
    fsize: dimensions of the filter

  Returns:
    g: *normalized* Gaussian filter
  """
  kenerl = np.zeros([fsize,fsize])
  center = fsize//2
  s = 2*(np.square(sigma))
  sum_val = 0
  for i in range(0, fsize):
    for j in range(0, fsize):
      x = i - center
      y = j - center
      exp = np.exp(-(x**2 + y**2) / s)
      kenerl[i,j] = exp/2*np.pi*np.square(sigma)
      sum_val += kenerl[i,j]
  sum_val = 1/sum_val
  return kenerl*sum_val
def createfilters():
  """
  Returns:
    fx, fy: filters as described in the problem assignment
  """
  derivative = 1/2*np.array([
    [1,0,-1],
    [1,0,-1],
    [1,0,-1]
  ])
  # gauss = ndimage.gaussian_filter1d(derivative, sigma = 0.9)
  gauss = gauss2d(sigma=0.9, fsize=3)
  fx = np.multiply(derivative, gauss)
  print(fx.shape)
  fy = fx.T
  print('fy shape is:')

  return fx,fy

def filterimage(I, fx, fy):
  """ Filter the image with the filters fx, fy.
  You may use the ndimage.convolve scipy-function.

  Args:
    I: a (H,W) numpy array storing image data
    fx, fy: filters

  Returns:
    Ix, Iy: images filtered by fx and fy respectively
  """

  Ix = ndimage.convolve(I, fx,mode='reflect')
  Iy = ndimage.convolve(I, fy,mode='reflect')
  return Ix, Iy

def detectedges(Ix, Iy, thr):
  """ Detects edges by applying a threshold on the image gradient magnitude.

  Args:
    Ix, Iy: filtered images
    thr: the threshold value

  Returns:
    edges: (H,W) array that contains the magnitude of the image gradient at edges and 0 otherwise
  """
  edges = np.hypot(Ix, Iy)
  edges = np.where( edges>thr, edges, 0)
  return edges


def nonmaxsupp(edges, Ix, Iy):
  """ Performs non-maximum suppression on an edge map.

  Args:
    edges: edge map containing the magnitude of the image gradient at edges and 0 otherwise
    Ix, Iy: filtered images

  Returns:
    edges2: edge map where non-maximum edges are suppressed
  """
  theta = np.arctan(Iy,Ix)
  M,N = Ix.shape
  edges2 = np.zeros((M,N))
  angle = theta * 180. / np.pi
  angle[angle < 0] += 180

  for i in range(1, M-1):
    for j in range(1, N-1):
      try:
        p = 1
        q = 1

        if (0<= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
          p = edges[i, j+1]
          q = edges[i, j-1]
        elif (22.5 <= angle[i,j] < 67.5):
          p = edges[i+1, j-1]
          q = edges[i-1, j+1]

        elif (67.5 <= angle[i,j] < 112.5):
          p = edges[i+1, j]
          q = edges[i-1, j]

        elif (112.5 <= angle[i,j] < 157.5):
          p = edges[i-1, j-1]
          q = edges[i+1, j+1]

        if (edges[i,j] >= p ) and (edges[i,j] >= q):
          edges2[i,j] = edges[i,j]
        else:
          edges2[i,j] = 0
      
      except IndexError as e:
        pass

  return edges2