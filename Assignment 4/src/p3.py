import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
import scipy.io as sio
import matplotlib.gridspec as gridspec
from epipolar_utils import *
import os

OUT = '../website/images/assignment-4'

'''
FACTORIZATION_METHOD The Tomasi and Kanade Factorization Method to determine
the 3D structure of the scene and the motion of the cameras.
Arguments:
    points_im1 - N points in the first image that match with points_im2
    points_im2 - N points in the second image that match with points_im1

    Both points_im1 and points_im2 are from the get_data_from_txt_file() method
Returns:
    structure - the structure matrix
    motion - the motion matrix
'''
def factorization_method(points_im1, points_im2):
    # Centre each image's points on their centroid
    x1 = points_im1[:, :2] - points_im1[:, :2].mean(axis=0)
    x2 = points_im2[:, :2] - points_im2[:, :2].mean(axis=0)

    # 2m x n measurement matrix, rows x1, y1, x2, y2
    D = np.vstack((x1.T, x2.T))
    U, S, Vt = np.linalg.svd(D)
    print("Singular values:", S)

    # Keep the rank 3 part and split sqrt(S) between motion and structure
    S3 = np.diag(np.sqrt(S[:3]))
    motion = U[:, :3] @ S3
    structure = S3 @ Vt[:3]
    return structure, motion

if __name__ == '__main__':
    for im_set in ['data/set1', 'data/set1_subset']:
        # Read in the data
        im1 = imread(im_set+'/image1.jpg')
        im2 = imread(im_set+'/image2.jpg')
        points_im1 = get_data_from_txt_file(im_set + '/pt_2D_1.txt')
        points_im2 = get_data_from_txt_file(im_set + '/pt_2D_2.txt')
        points_3d = get_data_from_txt_file(im_set + '/pt_3D.txt')
        assert (points_im1.shape == points_im2.shape)

        # Run the Factorization Method
        structure, motion = factorization_method(points_im1, points_im2)

        # Plot the structure
        fig = plt.figure()
        ax = fig.add_subplot(121, projection = '3d')
        scatter_3D_axis_equal(structure[0,:], structure[1,:], structure[2,:], ax)
        ax.set_title('Factorization Method')
        ax = fig.add_subplot(122, projection = '3d')
        scatter_3D_axis_equal(points_3d[:,0], points_3d[:,1], points_3d[:,2], ax)
        ax.set_title('Ground Truth')

        os.makedirs(OUT, exist_ok=True)
        fig.set_size_inches(12, 6)
        plt.savefig(f"{OUT}/p3_{im_set.split('/')[-1]}.png", bbox_inches='tight', dpi=100)

        plt.show()
