#Patches
import rasterio
import numpy as np
import os
import cv2

def crop(bounds, path=None, image=None):
    """Given a 4 pointed bounding box, crop sensor data
    Args:
        bounds: tuple of xmin, ymin, xmax, ymax, from shapely box.bounds -> geopandas.geometry.bounds
        path: path to sensor data, optional
        image: numpy array of sensor data, optional
    Returns:
        array: numpy array of cropped image
    """
    if path is None and image is None:
        raise ValueError("Must provide either path or image")   
    if image is None:
        image = cv2.imread(path)
    xmin, ymin, xmax, ymax = [int(x) for x in bounds]
    array = image[ymin:ymax, xmin:xmax]

    return array

def crop_predictions(df, root_dir, savedir):
    """Given a set of DeepForest predictions, crop the image and save to disk
    Args:
        df: dataframe of predictions
        root_dir: root directory of images
        savedir: directory to save crops
        Returns:
            crops: list of numpy arrays 
    """
    crops = []
    for index, row in df.iterrows():
        bounds = row["xmin"], row["ymin"], row["xmax"], row["ymax"] 
        array = crop(bounds, path=os.path.join(root_dir, row["image_path"]))
        crops.append(array)
    
    return crops