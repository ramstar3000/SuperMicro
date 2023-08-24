from numba import jit, cuda
import numpy as np
from skimage.measure import label


def getblob(stack, data, start_locations, initial_parameters, limits, chosen, stop=0, start=5, num_sigmas=3):
    """
    This function extracts the blobs from the stack and stores them in the data array. It finds the centroids of the blobs and stores them in the start_locations array.
    It then updates this back into the mutable numpy array chosen

    Parameters
    ----------
    stack : numpy.ndarray
        The stack of images. Format is (frame, x, y).

        ...

    num_sigmas : int, optional
        The number of sigmas to be used for thresholding. The default is 3.
        It uses mean + num_sigmas * std as the threshold.

    Empty Parameters (to be filled) 
    ----------

    data : numpy.ndarray
        The array that stores the box surround the blobs. Shape is (blob, 10, 10).
    start_locations : numpy.ndarray
        The array that stores the starting locations of the blobs box.  Shape is (blob, 2).
    initial_parameters : numpy.ndarray
        The array that stores the initial parameters of the blobs. [intensity, x, y, background, sigma]
    limits : Numba typed list
        The array that stores the starting and ending index of each frame in the data array.
    chosen : numpy.ndarray
        The array that stores the frame number of each blob. [frame number, total intensity, x, y, background, sigma] Rest is filled later
    stop : int, optional
        The number of frames to be ignored at the end of the stack. The default is 0.
    start : int, optional
        The number of frames to be ignored at the start of the stack. The default is 5.

    Returns
    -------
    None

     """

    for frame in range(start, len(stack) - stop):

        im = stack[frame].astype(np.float32)

        labels, nums = label((threshold(im, num_sigmas)), return_num=True)

        centroids = np.zeros((nums, 2), dtype=np.float32)
        get_coords(labels, nums, centroids)
        updateblobs(data, start_locations, limits, chosen,
                    centroids, im, initial_parameters)


@jit(target_backend='cuda', nopython=True, parallel=False)
def updateblobs(data, start_locations, limits, chosen, centroids, im, initial_parameters):
    """
    This function takes the centroids of the blobs and finds a 10x10 box around it. 
    It then stores the box in the data array and the starting location in the start_locations array. This runs on the GPU.

    Parameters
    ----------
    data : numpy.ndarray
        The array that stores the blobs.
    start_locations : numpy.ndarray
        The array that stores the starting locations of the blobs.
    limits : Numba typed list
        The array that stores the starting and ending index of each frame in the data array.
    chosen : numpy.ndarray
        The array that stores the frame number of each blob.
    centroids : numpy.ndarray
        The array that stores the centroids of the blobs.
    im : numpy.ndarray
        The image that the blobs are extracted from.
    initial_parameters : numpy.ndarray
        The array that stores the initial parameters of the blobs.

    Returns
    -------
    None

    """

    counter = limits[-1]

    for pred in range(len(centroids)):
        pred = centroids[pred]
        # y, x, width = pred
        box = im[int(pred[0]-5):int(pred[0]+5), int(pred[1]-5):int(pred[1]+5)]

        if (box.shape) == (10, 10):

            data[counter, :, :] = box
            start_locations[counter, :] = [int(pred[0]-5), int(pred[1]-5)]

        else:
            # Check if violates lower bound x
            if pred[0] - 5 < 0:

             # Check if violates lower bound y (both)
                if pred[1] - 5 < 0:

                    start_x = 0
                    start_y = 0

                    end_x = 10
                    end_y = 10

                    data[counter, :, :] = im[start_x:end_x, start_y:end_y]
                    start_locations[counter, :] = [start_x, start_y]

                # Check if violates upper bound y (both)
                elif pred[1] + 5 > im.shape[1]:

                    start_x = 0
                    start_y = im.shape[1] - 10

                    end_x = 10
                    end_y = im.shape[1]

                    data[counter, :, :] = im[start_x:end_x, start_y:end_y]
                    start_locations[counter, :] = [start_x, start_y]

                else:  # Just violates lower bound x

                    start_x = 0
                    start_y = int(pred[1]-5)

                    end_x = 10
                    end_y = int(pred[1]+5)

                    data[counter, :, :] = im[start_x:end_x, start_y:end_y]
                    start_locations[counter, :] = [start_x, start_y]

            # Check if violates upper bound x
            elif pred[0] + 5 > im.shape[0]:

                # Check if violates lower bound y
                if pred[1] - 5 < 0:

                    start_x = im.shape[0] - 10
                    start_y = 0

                    end_x = im.shape[0]
                    end_y = 10

                    data[counter, :, :] = im[start_x:end_x, start_y:end_y]
                    start_locations[counter, :] = [start_x, start_y]

                # Check if violates upper bound y
                elif pred[1] + 5 > im.shape[1]:

                    start_x = im.shape[0] - 10
                    start_y = im.shape[1] - 10

                    end_x = im.shape[0]
                    end_y = im.shape[1]

                    data[counter, :, :] = im[start_x:end_x, start_y:end_y]
                    start_locations[counter, :] = [start_x, start_y]

                else:
                    start_x = im.shape[0] - 10
                    start_y = int(pred[1]-5)

                    end_x = im.shape[0]
                    end_y = int(pred[1]+5)

                    data[counter, :, :] = im[start_x:end_x, start_y:end_y]
                    start_locations[counter, :] = [start_x, start_y]

            # Check if just violates lower bound y
            elif pred[1] - 5 < 0:

                start_x = int(pred[0]-5)
                start_y = 0

                end_x = int(pred[0]+5)
                end_y = 10

                data[counter, :, :] = im[start_x:end_x, start_y:end_y]
                start_locations[counter, :] = [start_x, start_y]

            # Check if just violates upper bound y
            elif pred[1] + 5 > im.shape[1]:

                start_x = int(pred[0]-5)
                start_y = im.shape[1] - 10

                end_x = int(pred[0]+5)
                end_y = im.shape[1]

                data[counter, :, :] = im[start_x:end_x, start_y:end_y]
                start_locations[counter, :] = [start_x, start_y]

        chosen[counter, 0] = len(limits) + 1
        counter += 1

    filtered2 = np.zeros((len(centroids), 5), dtype=np.float32)
    fill_filtered2(centroids, filtered2, im)

    initial_parameters[counter - filtered2.shape[0]: counter, :] = filtered2

    if limits[-1] == 0:
        limits[0] = counter
    else:
        limits.append(counter)


@jit(nopython=True, target_backend='cuda', fastmath=True)
def get_coords(labels, num, centroids):
    """

    This function takes the labels of the blobs and finds the centroids of the blobs. This runs on the GPU.

    Parameters
    ----------
    labels : numpy.ndarray
        The array that stores the labels of the blobs. This is a matrix of regions labelled with integers.
    num : int
        The number of blobs/regions
    centroids : numpy.ndarray
        The array that stores the centroids of the blobs. This is a 0 matrix and is filled in this function.

    Returns
    -------
    None

    """

    nz = np.nonzero(labels)
    coords = np.column_stack(nz)

    nzvalues2 = np.zeros(len(coords), dtype=np.int16)

    for i in range(len(coords)):
        nzvalues2[i] = labels[coords[i, 0], coords[i, 1]]

    for k in range(1, num + 1):

        temp = coords[nzvalues2 == k]
        xs = temp[:, 0]
        ys = temp[:, 1]

        centroids[k - 1, 0] = np.mean(xs)
        centroids[k - 1, 1] = np.mean(ys)


@jit(target_backend='cuda', nopython=True)
def threshold(im, num_sigmas):
    """
    This function is used to threshold the image. This runs on the GPU.

    Parameters
    ----------
    im : numpy.ndarray
        The image that the blobs are extracted from.

    Returns
    -------
    threshold_image: numpy.ndarray
        The thresholded image. Shape is same as input

    """

    m = np.mean(im)
    std = np.std(im)

    a = m + num_sigmas * std

    return im > a


@jit(target_backend='cuda', nopython=True)
def fill_filtered2(filtered, filtered2, im):
    """
    This is a helper function to fill the filtered2 array. This runs on the GPU.

    Parameters
    ----------
    filtered : numpy.ndarray
        The array that stores the centroids of the blobs.
    filtered2 : numpy.ndarray
        The array that stores the initial parameters of the blobs.
    im : numpy.ndarray
        The image that the blobs are extracted from.

    """

    for i in range(len(filtered)):
        # Originally had width data
        filtered2[i] = [im[int(filtered[i][0])]
                        [int(filtered[i][1])], 5, 5, 2, 0.25]
