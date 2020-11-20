import numpy as np
import matplotlib.pyplot as plt

def load_image(path):
    return plt.imread(path)


#
# Problem 1: Getting to know Python
#
from problem1 import *

def problem1():
    """Example code implementing the steps in Problem 1"""

    img = load_image("data/a1p1.png")
    display_image(img)

    save_as_npy("a1p1.npy", img)

    img1 = load_npy("a1p1.npy")
    display_image(img1)

    img2 = mirror_horizontal(img1)
    display_image(img2)

    display_images(img1, img2)


#
# Problem 2: Bayer Interpolation
#
from problem2 import *

def problem2():
    """Example code implementing the steps in Problem 2
    Note: uses display_image() from Problem 1"""

    data = loaddata("data/bayerdata.npy")
    display_image(data)
    r, g, b = separatechannels(data)
    display_image(r)
    display_image(g)
    display_image(b)

    img = assembleimage(r, g, b)
    display_image(img)

    img_interpolated = interpolate(r, g, b)
    display_image(img_interpolated)


#
# Problem 3: Projective Transformation
#
from problem3 import * 

def problem3():
    """Example code implementing the steps in Problem 3"""
    t = np.array([-27.1, -2.9, -3.2])
    principal_point = np.array([8, -10])
    focal_length = 8

    # model transformations
    T = gettranslation(t)
    Ry = getyrotation(135)
    Rx = getxrotation(-30)
    Rz = getzrotation(90)
    print(T)
    print(Ry)
    print(Rx)
    print(Rz)

    K = getcentralprojection(principal_point, focal_length)

    P,M = getfullprojection(T, Rx, Ry, Rz, K)
    print(P)
    print(M)

    points = loadpoints()
    displaypoints2d(points)

    z = loadz()
    Xt = invertprojection(K, points, z)

    Xh = inverttransformation(M, Xt)

    worldpoints = hom2cart(Xh)
    displaypoints3d(worldpoints)

    points2 = projectpoints(P, worldpoints)
    displaypoints2d(points2)

    plt.show()


#
# Problem 4: Image filtering and edge detection
#
from problem4 import *

def problem4():
    """Example code implementing the steps in Problem 4"""

    # load image
    img = plt.imread("data/a1p4.png")

    # create filters
    fx, fy = createfilters()

    # filter image
    imgx, imgy = filterimage(img, fx, fy)

    # show filter results
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.imshow(imgx, "gray", interpolation="none")
    ax1.set_title("x derivative")
    ax1.axis("off")
    ax2 = plt.subplot(122)
    ax2.imshow(imgy, "gray", interpolation="none")
    ax2.set_title("y derivative")
    ax2.axis("off")

    # show gradient magnitude
    plt.figure()
    plt.imshow(np.sqrt(imgx**2 + imgy**2), "gray", interpolation="none")
    plt.axis("off")
    plt.title("Derivative magnitude")

    # threshold derivative
    threshold = 0.01
    edges = detectedges(imgx,imgy,threshold)
    plt.figure()
    plt.imshow(edges > 0, "gray", interpolation="none")
    plt.axis("off")
    plt.title("Binary edges")

    # non maximum suppression
    edges2 = nonmaxsupp(edges,imgx,imgy)
    plt.figure()
    plt.imshow(edges2 > 0, "gray", interpolation="none")
    plt.axis("off")
    plt.title("Non-maximum suppression")

    plt.show()


if __name__ == "__main__":
    # problem1()
    # problem2()
    # problem3()
    problem4()
