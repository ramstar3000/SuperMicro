from sklearn.cluster import DBSCAN
import pandas as pd
import numpy as np
import os
from skimage.measure import regionprops_table
from numba import jit


def cluster_DBSCAN(field_name, file_dir, pixel_size, dimensions, eps=75.0, min_sample=5, scale=10, path_result_fid=None, df=None):
    """
    This function clusters the localisations in the file using DBSCAN. It then profiles the clusters and saves the profile file. It also saves the cluster localisation file.

    Parameters
    ----------
    field_name : str
        The name of the field.
    file_dir : str
        The directory of the file.
    pixel_size : int
        The pixel size of the localisations. The default is around 100.
    dimensions : tuple
        The dimensions of the image. The default is around (200, 200).
    eps : int, optional
        The maximum distance between two samples for one to be considered as in the neighborhood of the other. The default is from 75 to 100.
    min_sample : int, optional
        The number of samples in a neighborhood for a point to be considered as a core point. The default is 3 to 5
    scale : int, optional
        The magnification factor. The default is 10.
    path_result_fid : str, optional
        The directory to save the result files. The default is None and will be put in the same directory as the file in file_dir

    Returns
    -------
    None

    Files
    -----
    clusterProfile_{eps}_{min_sample}.csv : The profile of the clusters.

    """

    if df is None:

        if path_result_fid is None:
            path_result_fid = os.path.dirname(file_dir)  # Bad practice. Need to change

        df = pd.read_csv(file_dir)

    # The coordinates for localisations are in for TS results. This step unifies the unit to nm.
    df['X'] = df['x [nm]'] / pixel_size
    df['Y'] = df['y [nm]'] / pixel_size

    try:
        clustering = DBSCAN(eps=eps, min_samples=min_sample).fit(
            df[['x [nm]', 'y [nm]']])
    except ValueError:
        print('Not enough localisations for DBSCAN.')
        report = pd.DataFrame({
            'FoV': [field_name],
            'n_clusters': [0],
            'cluster_localisations': [0],
            'n_noise': [0],
            'total_localisations': [0]
        })
        return report

    labels = clustering.labels_
    n_clusters, n_noise = extract_labels(labels)

    # Label the localisations in the df with cluster ids

    labelled_df = df
    labelled_df['DBSCAN_label'] = labels  # Q

    # Remove localisations labelled as noise from the df
    cleaned_df = labelled_df
    cleaned_df = labelled_df[labelled_df.DBSCAN_label != -1]

    # Save cleaned cluster localisation file

    # Cluster profiling if cluster found
    if n_clusters != 0:

        # # Magnify the coordinates
        cleaned_df['X_mag'] = (cleaned_df['X'] * scale).astype('int16')
        cleaned_df['Y_mag'] = (cleaned_df['Y'] * scale).astype('int16')

        # Creates a dataframe contains all the pixel localisations
        placeholder = pd.DataFrame({
            'X_mag': np.tile(range(0, dimensions[0] * scale), dimensions[1] * scale),
            # Repeat x coordinates y times
            'Y_mag': np.repeat(range(0, dimensions[1] * scale), dimensions[0] * scale)
            # Repeat y coordinates x times
        })

        # These lines consume a lot a time. Need to find a way to speed up

        cluster_df = cleaned_df
        cluster_df['DBSCAN_label'] += 1  # Change label from 0-based to 1-based
        # Combine placeholder with actual dataframe
        cluster_df = pd.concat([cluster_df, placeholder],
                               axis=0, join='outer', sort=False)
        

        cluster_df3 = cluster_df.groupby(['Y_mag', 'X_mag'])['DBSCAN_label'].max().unstack(fill_value=0)
        cluster_img =  np.nan_to_num(cluster_df3.to_numpy(dtype=np.int16))  # convert pivot table to numpy array
        

        # TypeError as labels are not integers with Nans

        cluster_profile = regionprops_table(cluster_img, properties=[
                                            'label', 'area', 'major_axis_length', 'eccentricity'])  # Profile the aggregates

        cluster_profile = pd.DataFrame(cluster_profile)

        n_localisation = cleaned_df.groupby(['DBSCAN_label'])['id'].count()
        cluster_profile['n_localisation'] = n_localisation

        cluster_profile.columns = [
            'cluster_id', 'area',  'major_axis_length', 'eccentricity',  'n_localisation']

        # Save cluster profile file
        cluster_profile.to_csv(os.path.join(
            path_result_fid, field_name+'_clusterProfile_' + str(eps) + '_' + str(min_sample) + '.csv'))


@jit(target_backend='cuda', nopython=True)
def extract_labels(labels):
    """
    This function extracts the number of clusters and noise from the DBSCAN labels. This runs on the GPU.

    Parameters
    ----------
    labels : numpy.ndarray
        The array that stores the DBSCAN labels.

    Returns
    -------
    n_clusters : int
        The number of different clusters.
    n_noise : int
        The number of noisy clusters

    """

    n_noise = np.count_nonzero(labels == -1)
    n_clusters = len(set(labels))

    if n_noise != 0:
        n_clusters -= 1

    return n_clusters, n_noise


@jit(target_backend='cuda', nopython=True)
def localisation_precision(data, CameraPixelSize, rowslin, colslin, parameters, chosen):
    """
    This function calculates the localisation precision of the blob. This runs on the GPU.

    Parameters
    ----------
    particle : numpy.ndarray
        The array that stores the initial parameters of the blob. [intensity, x, y, sigma, background]
    data : numpy.ndarray
        The array that stores the box surround the blobs. Shape is (blob, 10, 10).
    CameraPixelSize : int
        The pixel size of the localisations. The default is around 100.
    rowslin : numpy.ndarray
        The array that stores the row number of the pixels in the box.
    colslin : numpy.ndarray
        The array that stores the column number of the pixels in the box.
    parameters : numpy.ndarray
        The array that stores the initial parameters of the blobs. [intensity, x, y, background, sigma]
    chosen : numpy.ndarray
        The array that stores the precision of each blob. [frame number, total intensity, x, y, background, sigma] Rest is filled later

    Returns
    -------
    chosen : numpy.ndarray
        The array that stores the precision of each blob. [frame number, total intensity, x, y, background, sigma, uncertainty] Rest is filled later

    """

    for particleID in range(len(parameters)):
        particle = parameters[particleID]

        if particle[3] == 0:
            continue

        inner = ((colslin - particle[2]) ** 2 +
                 ((rowslin - particle[1])) ** 2) / (2 * particle[3] ** 2)

        next = np.exp(-inner)
        background = np.std(data[particleID, :] -
                            (particle[4] + (particle[0] * next)))

        locPrec_1 = np.sqrt(((particle[3] * CameraPixelSize) ** 2) / np.sum(data[particleID, :]) + (CameraPixelSize ** 2) / (12 * np.sum(data[particleID, :])) + (
            4 * np.sqrt(np.pi) * ((particle[3] * CameraPixelSize) ** 3) * (background ** 2)) / ((CameraPixelSize) * (np.sum(data[particleID, :]) ** 2)))

        chosen[particleID, 5] = locPrec_1

        return chosen
