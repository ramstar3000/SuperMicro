from skimage import io
import numpy as np
import pygpufit.gpufit as gf
import pandas as pd
import time
import os
from datetime import datetime
from PIL import Image
from numba.typed import List as nlist
import warnings

from json.decoder import JSONDecodeError
from Estimator import getblob


import PySimpleGUI as Psg

from threading import Thread

from Drift import rcc_drift, dme_drift, DriftChoice, driftbead
from Reconstruction import reconstruction
from Grouping import grouping
from ClusterDetails import cluster_DBSCAN, localisation_precision
from segmentation import Sorter
from Functionals import gather_project_info, update_images, get_parameters
from UITools import get_directory, main_menu, error_message


def main(fname, outdir, name, outpath, window, estimate_fits_image=100,  tolerance=1e-2, max_number_iterations=15,
         start=0,  stop=0, drift=DriftChoice.NONE, num_beads=None, bead_dist=150 ,timebin=10, zoom=10, drift2=None, drift_graph=False, CameraPixelSize=100, SRPixelSize=10,
         roiRadius=5, cameraOffset=400, cameraEMGain=100, cameraQE=0.7, cameraConversionFactor=1, nframes=7000,
         sigmaMultiplier=1, max_sigma=160, max_localisation_precision=10, grouping_distance=25, threshholding_sigmas=4, eps=75.0, min_sample=2,
         reconstructiont=True, clusteringt=True, groupingt=True):

    s = time.time()

    stack = io.imread(fname, plugin='tifffile')

    dimensions = stack[0].shape
    size_x, size_y = dimensions

    stack, exclude = update_images(stack, directory, nframes, fname)

    print("Processing")

    num_images = len(stack) - stop - start

    # some sort of calulation to figure out average particles per picture
    n = estimate_fits_image * num_images
    data = np.zeros((n, 10, 10), dtype=np.float32)
    start_locations = np.zeros((n, 2), dtype=np.float32)
    initial_parameters = np.zeros((n, 5), dtype=np.float32)
    limits = nlist([0])
    chosen = np.zeros((n, 6), dtype=np.float32)

    a = time.time()


    #### 1 Fitting ####
    getblob(stack, data, start_locations, initial_parameters,
            limits, chosen, stop=stop, start=start, num_sigmas=threshholding_sigmas)

    b = time.time()

    window['output'].print(
        f"{b - a :.2f} s for thresholding | {limits[-1]} blobs | ", end="")

    number_fits = initial_parameters.shape[0]
    data = np.reshape(data, (number_fits, 100))


    parameters, states, _, _, _ = gf.fit(data, None, gf.ModelID.GAUSS_2D, initial_parameters,
                                         tolerance, max_number_iterations, None, gf.EstimatorID.LSE, None)


    

    # Change the units of intensity to photons
    parameters[:, 0] = (parameters[:, 0] - cameraOffset) / \
        (cameraEMGain / cameraQE) / cameraConversionFactor

    # Paramters: [Intensity, x, y, sigma, background]

    del initial_parameters
    del stack
    del limits

    ############################

    c = time.time()
    window['output'].print(f"{c - b :.2f} s for GpuFit | ", end="")
    

    #### 1.1 Localisation precision ####



    roiWidth = 2 * roiRadius + 1
    rowslin = np.tile(np.arange(1, roiWidth)[::-1], roiWidth - 1)
    colslin = np.tile(np.arange(1, roiWidth),
                      (roiWidth - 1, 1)).reshape(1, -1, order="F")
    chosen = localisation_precision(data, CameraPixelSize,
                           rowslin, colslin, parameters, chosen)

    chosen[:, 1:5] = parameters[:, :4]
    chosen[:, 2] = (chosen[:, 2] + start_locations[:, 1]) * CameraPixelSize
    chosen[:, 3] = (chosen[:, 3] + start_locations[:, 0]) * CameraPixelSize
    chosen[:, 4] = chosen[:, 4] * CameraPixelSize

    f = time.time()
    window['output'].print(f"{f - c :.2f} s for precision | ", end="")
    del data



    ############################

    #### 2 Grouping and Drift ####

    success = True


    p = time.time()
    if drift == DriftChoice.BEADS:
        success = driftbead(chosen, 100, CameraPixelSize,
                            num_images=num_images, num_beads=num_beads, bead_dist=bead_dist, outpath=outpath)


    filter2 = states == 0
    chosen = chosen[filter2]

    def filter1(x): return (0 <= x[:, 2]) & (x[:, 2] <= size_y * CameraPixelSize - 50) & (
        0 <= x[:, 3]) & (x[:, 3] <= size_x * CameraPixelSize - 50)

    chosen = chosen[filter1(chosen)]
    chosen = chosen[chosen[:, 5] < max_localisation_precision]
    chosen = chosen[chosen[:, 4] < max_sigma]

    # Filtering precision and sigma

    # Apply same filters t

    o = outpath if drift_graph else None

    if (drift == DriftChoice.RCC) or ((not success) & (drift2 == DriftChoice.RCC)):
        chosen = rcc_drift(chosen, CameraPixelSize, timebins=timebin,
                  zoom=zoom, outpath=o, sigma=chosen[:, 5])

    elif drift == DriftChoice.DME or ((not success) and (drift2 == DriftChoice.DME)):
        chosen = dme_drift(chosen, CameraPixelSize, timebins=timebin,
                  zoom=zoom, size_x=size_x, size_y=size_y, outpath=o)

    g = time.time()

    window['output'].print(f"{g - p :.2f} s for drift | ", end="")

    if groupingt:
        chosen = grouping(chosen, estimate_fits_image, max_dist=grouping_distance)

    ############################

    #### 3   Reconstruction   ####

    

    path_result_fid = os.path.dirname(outdir)


    if reconstructiont:
        zoom = int(CameraPixelSize / SRPixelSize)
        size = (int(size_x * CameraPixelSize / zoom),
                int(size_y * CameraPixelSize / zoom))
        imageSR_proc = np.zeros(size, dtype=np.float32)

        l = 10
        startrows = chosen[:, 2] / 10 - 5
        startrows[startrows < 0] = 0
        startrows = startrows.astype(int)

        startcols = chosen[:, 3] / 10 - 5
        startcols[startcols < 0] = 0
        startcols = startcols.astype(int)

        imageSR_proc =  reconstruction(chosen, startrows, startcols, l,
                    zoom, imageSR_proc, sigmaMultiplier)

        im = Image.fromarray(imageSR_proc)
        im.save(os.path.join(path_result_fid, f'{name}_reconstruction.tif'))

    c = time.time()
    window['output'].print(f"{c - g :.2f} s for reconstruction | ", end="")


    ############################
    

    x = pd.DataFrame(chosen)
    x.columns = ["Frame No", "Intensity", "x [nm]",
                 "y [nm]", "sigma [nm]", "Uncertainty"]
    x['id'] = x.index
    x.to_csv(outdir, float_format='%.5f', columns=[
             "id", "Frame No", "x [nm]", "y [nm]", "sigma [nm]", "Intensity", "Uncertainty"], index=False)
    d = time.time()

    window['output'].print(f"{d - c :.2f} s for saving | ", end="")

    ############################

    # Cluster the localisations

    if clusteringt:

        cluster_DBSCAN(name, outpath, pixel_size=CameraPixelSize,
                   dimensions=dimensions, path_result_fid=path_result_fid, eps=eps, min_sample=min_sample, df=x, scale = int(CameraPixelSize/SRPixelSize))

    ############################

    e = time.time()
    window['output'].print(
        f"{e - d :.2f} s for clustering | ", end=" ||")
    window['output'].print(f"{e-s : .2f}  total ")

    return exclude


def run_code(images, outdir, hyperparameters, window, directory):

    i = 0
    fnames = []

    num_files = hyperparameters["filenum"]

    while i < num_files: # Maybe -1 to make it work
 
        image = images[i]

        if image in fnames:
            i += 1
            continue

        name = os.path.splitext(os.path.basename(image))[0]
        outpath = os.path.join(outdir, f'{name}_result.csv')

        new_fnames = main(image, outpath, name, outpath, window, hyperparameters['estimate_fits_image'], hyperparameters['tolerance'],
                          hyperparameters['max_iterations'], hyperparameters['start'], hyperparameters['stop'], hyperparameters['drift'],
                          hyperparameters['num_beads'], hyperparameters['bead_dist'] ,hyperparameters['timebin'], hyperparameters['zoom'], hyperparameters['drift2'], hyperparameters['drift_graphs'],
                          hyperparameters['CameraPixelSize'], hyperparameters['SRPixelSize'] , hyperparameters['roiRadius'], hyperparameters['cameraOffset'],
                          hyperparameters['cameraEMGain'], hyperparameters['cameraQE'], hyperparameters['cameraConversionFactor'], hyperparameters['frames'],
                          1, hyperparameters['max_sigma'], hyperparameters['max_localisation_precision'], hyperparameters['grouping_distance'], hyperparameters['thresholding_sigmas'],
                          hyperparameters['eps'], hyperparameters['min_sample'], hyperparameters['reconstruction'], hyperparameters['clustering'], hyperparameters['grouping'])

        window['output'].print(f"Finished {i + 1 + len(new_fnames)} out of {num_files} files")

        i += 1

        fov_paths = gather_project_info(directory)

        to_add = set(list(fov_paths.values())) - set(images)
        fnames = fnames + new_fnames

        for image in to_add:

            images.append(image)

        

    window['output'].print(outpath)

    if hyperparameters['sorting']:

        if not hyperparameters['clustering']:
            error_message("Clustering must be on to sort the results", 2)
            exit()

        window['output'].print("Sorting the results")
        
        try:
            Sorter(outdir, directory)
        except FileNotFoundError as e:
            error_message(str(e) + "All other files have been saved", 2)
            exit()

    window['output'].print("Done, please close the window")


if __name__ == "__main__":

    # Remove warnings from numpy, activate for debugging
    np.seterr(divide='ignore', invalid='ignore')
    np.set_printoptions(suppress=True)
    pd.options.mode.chained_assignment = None

    # Turn off all warnings
    warnings.filterwarnings("ignore")

    directory = get_directory()

    top_directory = os.path.dirname(directory)
    bottom = os.path.basename(directory)
    fov_paths = gather_project_info(directory)
    images = list(fov_paths.values())

    # Create the output directory
    t = datetime.now().strftime("%Y-%m-%d_%H-%M")  # Add second if needed
    outdir = os.path.join(top_directory, f'{bottom} Results {t}')
    os.mkdir(outdir)

    try:
        hyperparameters = get_parameters(directory)
    except JSONDecodeError as e:
        error_message(str(e), 1)
        exit()
    except FileNotFoundError as e:
        error_message(str(e), 2)
        exit()

    window = main_menu()

    window['output'].print(f"Directory: {directory}")

    # Create 2 threads to run the code one for the GUI and one for the code

    a = Thread(target=run_code, args=(
        images, outdir, hyperparameters, window, directory))
    flag = True

    while True:

        event, values = window.read(timeout=1)

        if flag:
            a.start()
            flag = False

        if event == Psg.WIN_CLOSED or event == "OK" or event == "Cancel":
            break
