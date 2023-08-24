from numba import jit
import numpy as np




@jit(target_backend='cuda', nopython=True)
def grouping(chosen, estimate_fits_image, max_dist=25, dist=4):
    """
    This function groups the localisations into blobs. It then updates the chosen array with the frame number of each blob. Marks grouped particles for removal with -1.

    Parameters
    ----------
    chosen : numpy.ndarray [frame number, total intensity, x, y, background, sigma]
        The array that stores the frame number of each blob.
    estimate_fits_image : int
        The number of frames in the stack.
    max_dist : int, optional
        The maximum distance between two blobs to be considered as in the neighborhood of the other. The default is 25.
    dist : int, optional
        The number of frames to be looked ahead. The default is 4.

    Returns
    -------
    chosen : numpy.ndarray
        The array that stores the frame number of each blob.



    """

    count = 0

    for particleID in range(len(chosen) - dist * estimate_fits_image - 1):
        particle = chosen[particleID, :]

        if particle[0] == -1:
            break

        potential_limits = (
            particleID + 5, estimate_fits_image * dist + particleID)

        for p in range(potential_limits[0], potential_limits[1]):
            potential = chosen[p]

            if potential[0] == particle[0] and particle[2] == potential[2] and particle[3] == potential[3]:
                # Same particle
                chosen[p, 0] = -1

            # Make equality <= should be -1?
            if potential[0] <= particle[0] or potential[0] == 0 or particle[0] == 0:
                continue

            if (particle[2] - potential[2]) ** 2 + (particle[3] - potential[3]) ** 2 < max_dist ** 2:

                update(chosen, p, particle[1] + potential[1], particle[2] +
                       potential[2], particle[3] + potential[3], potential[0])
                count += 1
                chosen[particleID, 0] = -1

            else:
                pass

    chosen = g_filter(chosen)
    return chosen


@jit(target_backend='cuda', nopython=True)
def update(chosen, p, m1, m2, m3, frame):

    chosen[p, 0] = frame
    chosen[p, 1] = m1 / 2
    chosen[p, 2] = m2 / 2
    chosen[p, 3] = m3 / 2


@jit(nopython=True, target_backend='cuda')
def g_filter(chosen):

    """
    This function removes the grouped particles from the chosen array.

    Parameters
    ----------
    chosen : numpy.ndarray [frame number, total intensity, x, y, background, sigma]
        The array that stores the frame number of each blob.
    
    Returns
    -------
    chosen : numpy.ndarray
        The array that stores the frame number of each blob.
    """

    chosen2 = chosen[chosen[:, 0] != -1]
    chosen = chosen2

    return (chosen)