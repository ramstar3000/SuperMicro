from numba import jit
import numpy as np


@jit(target_backend='cuda', nopython=True)
def reconstruction(localisations, startrows, startcols,  l, zoom, imageSR_proc,sigmaMultiplier = 1):
    """
    This function reconstructs the image from the localisations. It mutates the imageSR_proc array to store the reconstructed image.

    Parameters
    ----------
    localisations : numpy.ndarray [frame number, total intensity, x, y, background, sigma]
        The array that stores all the localisations.
    startrows : numpy.ndarray
        The array that stores the starting row of each blob.
    startcols : numpy.ndarray
        The array that stores the starting column of each blob.
    l : int
        The length of the blob fed into gaussian.
    zoom : int
        The zoom factor normally 10?
    imageSR_proc : numpy.ndarray
        The array that stores the reconstructed image. In microns.
    """


    for i in range(len(localisations)):  # Can be made dynamic? Test

        particle = localisations[i]

        # 

        startr = startrows[i]
        startc = startcols[i]

        a = meshgrid(np.arange(startr, startr + l),
                     np.arange(startc, startc + l))

        # Here use 10 in the denominator to convert to nm
        calcf = particle[1] * np.exp(-(((a[0] - particle[2] / zoom) ** 2 + (
            a[1] - particle[3] / zoom) ** 2)) / (2 * (particle[5] * sigmaMultiplier / zoom) ** 2))

        calcf = np.clip(calcf, 0, None)

        imageSR_proc[startc: startc + l, startr: startr + l] += calcf

    
    return imageSR_proc


@jit(target_backend='cuda', nopython=True)
def meshgrid(y, x):

    """
    Helper function to create a meshgrid as numpy's meshgrid is not supported by numba.

    Parameters
    ----------
    y : numpy.ndarray
    x : numpy.ndarray

    Returns
    -------
    [yy, xx] : list
    
    """

    xx = np.empty(shape=(x.size, y.size), dtype=x.dtype)
    yy = np.empty(shape=(x.size, y.size), dtype=y.dtype)

    for i in range(0, x.size):
        for j in range(0, y.size):
            xx[i, j] = x[i]
            yy[i, j] = y[j]

    return [yy, xx]
