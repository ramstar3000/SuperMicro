from numba import jit, cuda
from dme.dme import dme_estimate
from dme.rcc import rcc3D, rcc
from dme.native_api import NativeAPI
from matplotlib import pyplot as plt

import numpy as np

# import a spline fitting function
from scipy.interpolate import UnivariateSpline


class DriftChoice:
    """
    This class is an enum class that stores the different drift correction methods.

    Attributes
    ----------
    NONE
        No drift correction.
    RCC
        RCC drift correction.
    DME
        DME drift correction.
    BEADS
        Beaded drift correction.
    """
    NONE = 0
    RCC = 1
    DME = 2
    BEADS = 3


@jit(target_backend='cuda', nopython=True)
def change(chosen, estimate_drift_rcc, CameraPixelSize):
    """
    This function changes the coordinates of the localisations based on the drift correction.

    Parameters
    ----------
    chosen : numpy.ndarray
        The array that stores the localisations.
    estimate_drift_rcc : numpy.ndarray
        The array that stores the drift correction. This is in pixels
    CameraPixelSize : int
        The pixel size of the camera.

    Mutates
    -------
    chosen : numpy.ndarray
        The array that stores the localisations by undrifting them



    """

    for i in range(len(chosen)):

        chosen[i][2] = chosen[i][2] - \
            estimate_drift_rcc[int(chosen[i][0])][0] * CameraPixelSize
        chosen[i][3] = chosen[i][3] - \
            estimate_drift_rcc[int(chosen[i][0])][1] * CameraPixelSize


def rcc_drift(chosen, CameraPixelSize, timebins=10, zoom=2, outpath=None, sigma=1):
    """
    This function performs RCC drift correction on the localisations.

    Parameters
    ----------
    chosen : numpy.ndarray
        The array that stores the localisations.
    CameraPixelSize : int
        The pixel size of the camera.
    timebins : int, optional
        The number of frames to be binned together. The default is 10.
    zoom : int, optional
        The zoom factor for RCC. The default is 1.
    printout : bool, optional
        Whether to print the drift correction. The default is False. For debugging purposes.
    outpath : str, optional
        The path to save the drift correction plot. The default is None.
    sigma : numpy.ndarray, optional

    Returns
    -------
    chosen : numpy.ndarray
        The array that stores the localisations by undrifting them


    """

    localisations = np.zeros((len(chosen), 2))  # 3 -> 2
    localisations[:, :2] = chosen[:, 2:4] / CameraPixelSize

    framenum = chosen[:, 0]
    framenum = framenum.astype(int)


    estimate_drift_rcc = rcc(localisations, framenum, timebins=timebins,
                             zoom=zoom, sigma=2, dll=NativeAPI(True)) # Sigma was set to 2 before

    estimates = estimate_drift_rcc[1]
    estimate_drift_rcc = estimate_drift_rcc[0]

    estimate_drift_rcc = (estimate_drift_rcc - estimate_drift_rcc[0])

    if outpath is not None:

        plt.title(f'Drift Correction RCC, zoom = {zoom} timebins = {timebins}')

        plt.plot(estimate_drift_rcc[:, 0], label='x')
        plt.plot(estimate_drift_rcc[:, 1], label='y')

        plt.scatter(estimates[:, 2], estimates[:, 0], s=1, c='r')
        plt.scatter(estimates[:, 2], estimates[:, 1], s=1, c='r')

        plt.savefig(outpath + 'driftrcc.png')

        plt.clf()

    change(chosen, estimate_drift_rcc, CameraPixelSize)

    return chosen


def dme_drift(chosen, CameraPixelSize, timebins=10, zoom=1, size_x=100, size_y=100, outpath=None):
    """
    This function performs DME drift correction on the localisations.

    Parameters
    ----------
    chosen : numpy.ndarray
        The array that stores the localisations.
    CameraPixelSize : int
        The pixel size of the camera.
    timebins : int, optional 
        The number of frames to be binned together. The default is 10.
    zoom : int, optional
        The zoom factor for RCC. The default is 1.
    size_x : int, optional
        The size of the image in the x direction. The default is 100.
    size_y : int, optional
        The size of the image in the y direction. The default is 100.
    outpath : str, optional
        The path to save the drift correction plot. The default is None.
    
        
    Returns
    -------
    chosen : numpy.ndarray
        The array that stores the localisations by undrifting them
    """

    localisations = chosen[:, 2:4] / CameraPixelSize

    framenum = chosen[:, 0]
    framenum = framenum.astype(int)

    # CRLB needs x and y

    crlb = np.ones((len(chosen), 2), dtype=np.float32) * \
        np.tile((chosen[:, 5] / CameraPixelSize), (2, 1)).T
    # crlb = np.array([chosen[:, 5] / CameraPixelSize, chosen[:, 5] / CameraPixelSize])

    estimated_drift = dme_estimate(
        localisations,
        framenum,
        imgshape=[size_x, size_y],
        perSpotCRLB=True,
        crlb=crlb,  # 20 nm precision
        framesperbin=timebins,  # note that small frames per bin use many more iterations
        coarseFramesPerBin=200,
        coarseSigma=[0.5, 0.5],
        useCuda=True,
        useDebugLibrary=False,
        rccZoom=zoom,
    )

    if outpath is not None:

        plt.title(f'Drift Correction DME, zoom = {zoom} timebins = {timebins}')

        plt.plot(estimated_drift[:, 0], label='x')
        plt.plot(estimated_drift[:, 1], label='y')

        plt.savefig(outpath + 'driftdme.png')

        plt.clf()

    estimated_drift = estimated_drift - estimated_drift[0]

    change(chosen, estimated_drift, CameraPixelSize)

    return chosen



@jit(target_backend='cuda', nopython=True)
def beads(chosen, num_beads=2, num_images=10000, max_dist=200):
    """
    This function performs beaded drift correction on the localisations.

    Parameters
    ----------
    chosen : numpy.ndarray
        The array that stores the localisations.
    num_beads : int, optional
        The number of beads to be used. The default is 2.
    num_images : int, optional
        The number of images to be used. The default is 10000.
    max_dist : int, optional
        The maximum distance between bead in subsequent frames. The default is 200.

    Returns
    -------
    beadx : numpy.ndarray
        The array that stores the x coordinates of the beads.
    beady : numpy.ndarray
        The array that stores the y coordinates of the beads.

    """

    # Looking for particles on every frame

    initials = chosen[:800]  # Set to minimum frame number
    chains = np.zeros(len(initials), dtype=np.int16)

    old_access = np.zeros((len(chosen)))
    new_access = np.zeros((len(chosen)))

    beadx = np.zeros((num_beads, num_images + 2), dtype=np.float32)
    beady = np.zeros((num_beads, num_images + 2), dtype=np.float32)
    
    lengths = [0 for i in range(num_beads)]
    
    b = 0


    for p in range(len(initials)):
        particle = initials[p]

        second = initials[p]

        beadx[b][int(particle[0])] = particle[2]
        beady[b][int(particle[0])] = particle[3]

        for i in range(p + 1, len(chosen)):

            if old_access[i] == 1:
                continue

            potential = chosen[i][:]

            if (particle[2] - potential[2]) ** 2 + (particle[3] - potential[3]) ** 2 < max_dist ** 2:

                if particle[0] != potential[0]:
                    particle = potential
                    chains[p] += 1

                    beadx[b][int(particle[0])] = particle[2]
                    beady[b, int(particle[0])] = particle[3]

                    new_access[i] = 1

                    continue

            elif (potential[2] - second[2]) ** 2 + (potential[3] - second[3]) ** 2 < max_dist ** 2:

                if potential[0] != second[0]:
                    particle = second
                    chains[p] += 1

                    beadx[b][int(particle[0])] = particle[2]
                    beady[b, int(particle[0])] = particle[3]

                    new_access[i] = 1

                    continue

            if (i - p > 35 * (len(chosen) / num_images)) and (chains[p] < 18):
                break

            if (i - p > 50 * (len(chosen) / num_images)) and (chains[p] < 23):
                break

        if chains[p] > num_images / 1.3:
            lengths[b] =  (chains[p])
            b += 1
            old_access = new_access

            if b >= num_beads / 2:
                break

        else:
            beadx[b] *= 0  # Can be simplified somehow
            beady[b] *= 0
            new_access = old_access

       
        
    return b, beadx, beady, lengths


def driftbead(chosen, kernel_size, CameraPixelSize, num_images, num_beads=2, bead_dist = 150 ,outpath=None):
    """
    This function performs beaded drift correction on the localisations.

    Parameters
    ----------
    chosen : numpy.ndarray
        The array that stores the localisations.
    kernel_size : int, optional
        The size of the kernel to be used for convolution. The default is 100.
    CameraPixelSize : int
        The pixel size of the camera
    num_images : int
        The number of images to be used
    num_beads : int, optional
        The number of beads to be used. The default is 2.
    bead_dist : int, optional
        The maximum distance between bead in subsequent frames. The default is 150.
    outpath : str, optional
        The path to save the drift correction plot. The default is None.
    



    Mutates
    -------
    chosen : numpy.ndarray
        The array that stores the localisations by undrifting them

    """

    b, driftx, drifty, lengths = beads(chosen, num_beads * 4, num_images, bead_dist)
    
    if b <= num_beads/2 :
                
        return False
    
    else:

        temp = [lengths.index(f) for f in sorted(lengths, reverse=True)]
        temp = np.array(temp[:num_beads])


        driftx = driftx[temp]
        drifty = drifty[temp]


    

        

        
        
    
    

    drift_average_x, drift_average_y, kernel = preprocess(
        driftx, drifty, kernel_size)

    # Find the driftx and drifty for each frame

    average_x, average_y = convolve(drift_average_x, drift_average_y, kernel)

    # Average x is measured 

    if outpath is not None:


        plt.scatter(np.arange(len(average_x)), average_x, )
        plt.scatter(np.arange(len(average_y)), average_y)
        plt.savefig(outpath + 'driftbead.png')
        
        plt.clf()
        

    # Already in nm so no need to scale ahain
    change(chosen, np.array([average_x, average_y]).T, CameraPixelSize=1)

    return True


# @jit(target_backend='cuda', nopython=True)
def preprocess(driftx, drifty, kernel_size):

    # May not run on GPU, can be pushed if needed
    """
    This function preprocesses the drift correction by removing outliers and smoothing the drift correction. This uses a moving average filter via a convolution.
    Runs on the GPU.

    Parameters
    ----------
    driftx : numpy.ndarray
        The array that stores the x coordinates of the drift correction.
    drifty : numpy.ndarray
        The array that stores the y coordinates of the drift correction.
    kernel_size : int, optional
        The size of the kernel to be used for convolution. The default is 100.

    Returns
    -------
    drift_average_x : numpy.ndarray
        The array that stores the x coordinates of the drift correction after processing.
    drift_average_y : numpy.ndarray
        The array that stores the y coordinates of the drift correction after processing.

    """

    # Drift is the mean of the non-zero drifts

    drift_average_x = sum(driftx) / (np.count_nonzero(driftx, axis=0) + 1)
    drift_average_y = sum(drifty) / (np.count_nonzero(drifty, axis=0) + 1)

    s = 2
    drift_average_x = drift_average_x[s:]
    drift_average_y = drift_average_y[s:]

    drift_average_x = drift_average_x - drift_average_x[0]
    drift_average_y = drift_average_y - drift_average_y[0]

    for i in range(1, len(drift_average_x)):

        if abs(drift_average_x[i] - drift_average_x[i - 1]) > 100:
            drift_average_x[i] = drift_average_x[i - 1]
        
        if abs(drift_average_y[i] - drift_average_y[i - 1]) > 100:
            drift_average_y[i] = drift_average_y[i - 1]


    kernel = np.ones((kernel_size,)) / kernel_size

    return drift_average_x, drift_average_y, kernel




@jit(target_backend='cuda', nopython=True)
def convolve(drift_average_x, drift_average_y, kernel):
    """ 
    This function performs a convolution on the drift correction. This uses a moving average filter via a convolution.

    Parameters
    ----------
    drift_average_x : numpy.ndarray
        The array that stores the x coordinates of the drift correction after processing.
    drift_average_y : numpy.ndarray
        The array that stores the y coordinates of the drift correction after processing.

    Returns
    -------
    average_x : numpy.ndarray
        The array that stores the x coordinates of the drift correction after convolution.
    average_y : numpy.ndarray
        The array that stores the y coordinates of the drift correction after convolution.

    """
    
    for i in range(3, len(drift_average_x)-5):

            drift_average_x[i] = np.sum(drift_average_x[i - 5: i + 5]) / 10
            drift_average_y[i] = np.sum(drift_average_y[i - 5: i + 5]) / 10
            
    for i in range( len(drift_average_x)):

            drift_average_x[i] = np.sum(drift_average_x[i - 2: i + 2]) / 4
            drift_average_y[i] = np.sum(drift_average_y[i - 2: i + 2]) / 4
            
    average_x = drift_average_x.copy()
    average_y = drift_average_y.copy()           
            

    for i in range( len(drift_average_x)-100):

            average_x[i] = np.sum(drift_average_x[i - 50: i + 50]) / 100

            average_y[i] = np.sum(drift_average_y[i - 50: i + 50]) / 100


    for i in range(len(drift_average_x)-120, len(drift_average_x)):

            average_x[i] = np.sum(drift_average_x[i - 2: i + 4]) / 6

            average_y[i] = np.sum(drift_average_y[i - 2: i + 4]) / 6
        


    return average_x, average_y
