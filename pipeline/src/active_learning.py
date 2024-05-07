import glob
import os
import random
from src import model
import pandas as pd

def choose_images(evaluation, image_dir, strategy, n=10, patch_size=512, patch_overlap=0.1, min_score=0.5, m=None, dask_client=None):
    """Choose images to annotate.
    Args:
        evaluation (dict): A dictionary of evaluation metrics.
        image_dir (str): The path to a directory of images.
        strategy (str): The strategy for choosing images. Available strategies are:
            - "random": Choose images randomly from the pool.
            - "most-detections": Choose images with the most detections based on predictions.
        n (int, optional): The number of images to choose. Defaults to 10.
    Returns:
        list: A list of image paths.
    """
    pool = glob.glob(os.path.join(image_dir,"*")) # Get all images in the data directory
    if strategy=="random":
        chosen_images = random.sample(pool, n)
        return chosen_images    
    elif strategy=="most-detections":
        # Predict all images
        if m is None:
            raise ValueError("A model is required for the 'most-detections' strategy.")
        preannotations = model.predict(m, pool, patch_size=patch_size, patch_overlap=patch_overlap, min_score=min_score, client=dask_client)
        preannotations = pd.concat(preannotations)
        # Sort images by total number of predictions
        chosen_images = preannotations.groupby("image_path").size().sort_values(ascending=False).head(n).index.tolist()
        
        # Get full path
        chosen_images = [os.path.join(image_dir, image) for image in chosen_images]

    return chosen_images