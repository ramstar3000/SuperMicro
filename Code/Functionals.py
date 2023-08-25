import os
import re
import time
from skimage import io
import numpy as np
from json import load
from Drift import DriftChoice


def get_parameters(directory):
    '''
    This function reads the parameters.txt file and returns a dictionary of the hyperparameters.

    Parameters
    ----------
    directory : str
        The directory containing the images.

    Returns
    -------
    hyperparameters : dict
        A dictionary of the hyperparameters.    
    '''

    with open(directory + "/parameters.txt", "r") as f:
        hyperparameters = load(f)

        a = hyperparameters["drift"]
        if a == "RCC":
            hyperparameters['drift'] = DriftChoice.RCC
            hyperparameters['num_beads'] = None
            hyperparameters['drift2'] = None
            hyperparameters['bead_dist'] = None
        elif a == "DME":
            hyperparameters['drift'] = DriftChoice.DME
            hyperparameters['num_beads'] = None
            hyperparameters['drift2'] = None
            hyperparameters['bead_dist'] = None

        elif a == "BEADS":
            hyperparameters['drift'] = DriftChoice.BEADS

            hyperparameters['drift2'] = DriftChoice.DME if hyperparameters['drift2'] == "DME" else DriftChoice.RCC

        else:
            hyperparameters['drift'] = DriftChoice.NONE
            hyperparameters['num_beads'] = None
            hyperparameters['drift2'] = None

    return hyperparameters


def update_images(stack, directory, nframes, fname):
    '''
    This function updates the stack with new images that have been added to the directory.

    Parameters
    ----------
    stack : numpy.ndarray
        The stack of images.
    directory : str
        The directory containing the images.
    nframes : int
        The number of frames in the stack.
    fname : str
        The name of the first image in the stack.

    Returns
    -------
    stack : numpy.ndarray
        The updated stack of images.
    exclude : list
        A list of the images that were added to the stack.

    '''

    fnames = []
    exclude = []

    while len(stack) < nframes:

        # Check if new file with same initial
        fov_paths = gather_project_info(directory, p=False)

        images = fov_paths.values()
        startf, extension = os.path.splitext(fname)
        for image in images:
            if f'{startf[:-1]}{len(stack)+1}{extension}' == image:
                if image not in fnames:
                    exclude.append(image)

        if len(exclude) == 0:
            time.sleep(9)
            continue

        stack = io.imread(fname, plugin='tifffile')

        for fnam in exclude:
            stackt = io.imread(fnam, plugin='tifffile')

            stack = np.concatenate((stack, stackt), axis=0)

        if len(stack) != nframes:

            time.sleep(1)

    return stack, exclude


def gather_project_info(path_data_main, p=False):
    '''
    This function gathers the information about the images in the directory.

    Parameters
    ----------
    path_data_main : str
        The directory containing the images.
    p : bool, optional
        Whether to print the names of the images. The default is False.

    Returns
    -------
    fov_paths : dict
        A dictionary of the images in the directory.
    wells : dict
        A dictionary of the wells in the directory.
    '''

    fov_paths = {}  # dict - FoV name: path to the corresponding image

    for root, dirs, files in os.walk(path_data_main):
        for file in files:
            if file.endswith(".tiff"):  # or file.endswith(".tif") :  # Maybe tif
                if p:
                    print(file, end=" | ")
                try:
                    pos = re.findall(r"X\d+Y\d+R\d+W\d+C\d+", file)[-1]
                except IndexError:
                    try:
                        pos = re.findall(r'X\d+Y\d+R\d+W\d+', file)[-1]
                    except IndexError:
                        raise IndexError(
                            'Error in the naming system of the images. Please make sure the image names contain coordinate in form of XnYnRnWnCn or XnYnRnWn.')
                fov_paths[pos] = os.path.join(root, file)

    return fov_paths
