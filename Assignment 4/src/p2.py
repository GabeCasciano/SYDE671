import numpy as np
import matplotlib.pyplot as plt
from p1 import *
from epipolar_utils import *

'''
COMPUTE_EPIPOLE computes the epipole e in homogenous coordinates
given the fundamental matrix
Arguments:
    F - the Fundamental matrix solved for with normalized_eight_point_alg(points1, points2)

Returns:
    epipole - the homogenous coordinates [x y 1] of the epipole in the image
'''
def compute_epipole(F):
    # e is the null vector of F^T: last right singular vector
    _, _, Vt = np.linalg.svd(F.T)
    e = Vt[-1]
    return e / e[2]

'''
COMPUTE_H computes a homography to map an epipole to infinity along the horizontal axis 
Arguments:
    e - the epipole
    im2 - the image
Returns:
    H - homography matrix
'''
def compute_H(e, im):
    h, w = im.shape[:2]
    T = np.array([[1, 0, -w/2], [0, 1, -h/2], [0, 0, 1]])

    # Rotate the translated epipole onto the x axis
    e_t = T @ e
    e_t = e_t / e_t[2]
    e1, e2 = e_t[0], e_t[1]
    r = np.sqrt(e1**2 + e2**2)
    a = 1 if e1 >= 0 else -1
    R = np.array([[a*e1/r, a*e2/r, 0], [-a*e2/r, a*e1/r, 0], [0, 0, 1]])

    # Send (f, 0, 1) to infinity
    f = (R @ e_t)[0]
    G = np.array([[1, 0, 0], [0, 1, 0], [-1/f, 0, 1]])
    return np.linalg.inv(T) @ G @ R @ T

'''
COMPUTE_MATCHING_HOMOGRAPHIES determines homographies H1 and H2 such that they
rectify a pair of images
Arguments:
    e2 - the second epipole
    F - the Fundamental matrix
    im2 - the second image
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1
Returns:
    H1 - the homography associated with the first image
    H2 - the homography associated with the second image
'''
def compute_matching_homographies(e2, F, im2, points1, points2):
    H2 = compute_H(e2, im2)

    # M = [e]x F + e v^T with v = (1, 1, 1)
    ex = np.array([[0, -e2[2], e2[1]], [e2[2], 0, -e2[0]], [-e2[1], e2[0], 0]])
    M = ex @ F + np.outer(e2, np.ones(3))

    p_hat = (H2 @ M @ points1.T).T
    p_hat /= p_hat[:, 2:]
    p_hat2 = (H2 @ points2.T).T
    p_hat2 /= p_hat2[:, 2:]

    # Least squares W a = b for the first row of HA
    a, *_ = np.linalg.lstsq(p_hat, p_hat2[:, 0], rcond=None)
    HA = np.array([a, [0, 1, 0], [0, 0, 1]])
    H1 = HA @ H2 @ M
    return H1, H2

if __name__ == '__main__':
    # Read in the data
    im_set = 'data/set1'
    im1 = imread(im_set+'/image1.jpg')
    im2 = imread(im_set+'/image2.jpg')
    points1 = get_data_from_txt_file(im_set+'/pt_2D_1.txt')
    points2 = get_data_from_txt_file(im_set+'/pt_2D_2.txt')
    assert (points1.shape == points2.shape)

    F = normalized_eight_point_alg(points1, points2)
    # F is such that such that (points2)^T * F * points1 = 0, so e1 is e' and e2 is e
    e1 = compute_epipole(F.T)
    e2 = compute_epipole(F)
    print("e1", e1)
    print("e2", e2)

    # Find the homographies needed to rectify the pair of images
    H1, H2 = compute_matching_homographies(e2, F, im2, points1, points2)
    print('')

    # Transforming the images by the homographies
    new_points1 = H1.dot(points1.T)
    new_points2 = H2.dot(points2.T)
    new_points1 /= new_points1[2,:]
    new_points2 /= new_points2[2,:]
    new_points1 = new_points1.T
    new_points2 = new_points2.T
    rectified_im1, offset1 = compute_rectified_image(im1, H1)
    rectified_im2, offset2 = compute_rectified_image(im2, H2)
    new_points1 -= offset1 + (0,)
    new_points2 -= offset2 + (0,)

    # Plotting the image
    F_new = normalized_eight_point_alg(new_points1, new_points2)
    print("H1\n", H1)
    print("H2\n", H2)
    plot_epipolar_lines_on_images(new_points1, new_points2, rectified_im1, rectified_im2, F_new)
    plt.savefig(f'{OUT}/p2_rectified.png', bbox_inches='tight', dpi=100)
    plt.show()
