import glob
import os
import random
from src import model
import dask.array as da
import pandas as pd

def choose_images(evaluation, image_dir, strategy, n=10, patch_size=512, patch_overlap=0.1, min_score=0.5, m=None, model_path=None, dask_client=None):
    """Choose images to annotate.
    Args:
        evaluation (dict): A dictionary of evaluation metrics.
        image_dir (str): The path to a directory of images.
        strategy (str): The strategy for choosing images. Available strategies are:
            - "random": Choose images randomly from the pool.
            - "most-detections": Choose images with the most detections based on predictions.
        n (int, optional): The number of images to choose. Defaults to 10.
        dask_client (dask.distributed.Client, optional): A Dask client for parallel processing. Defaults to None.
        patch_size (int, optional): The size of the image patches to predict on. Defaults to 512.
        patch_overlap (float, optional): The amount of overlap between image patches. Defaults to 0.1.
        min_score (float, optional): The minimum score for a prediction to be included. Defaults to 0.5.
        m (main.deepforest, optional): A trained deepforest model. Defaults to None. 
        model_path (str, optional): The path to the model checkpoint file. Defaults to None. Only used in combination with dask
    Returns:
        list: A list of image paths.
    """
    pool = glob.glob(os.path.join(image_dir,"*")) # Get all images in the data directory
    if strategy=="random":
        chosen_images = random.sample(pool, n)
        return chosen_images    
    elif strategy=="most-detections":
        # Predict all images
        if model_path is None:
            raise ValueError("A model is required for the 'most-detections' strategy.")
        if dask_client:
            # load model on each client
            def update_sys_path():
                import sys
                sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            dask_client.run(update_sys_path)

            # Load model on each client
            dask_pool = da.from_array(pool, chunks=len(pool)//len(dask_client.ncores()))
            blocks = dask_pool.to_delayed().ravel()
            block_futures = []
            for block in blocks:
                block_future = dask_client.submit(model.predict,image_paths=block.compute(), patch_size=patch_size, patch_overlap=patch_overlap, min_score=min_score, model_path=model_path)
                block_futures.append(block_future)
            # Get results
            dask_results = []
            for block_result in block_futures:
                block_result = block_result.result()
                dask_results.append(pd.concat(block_result))
            preannotations = pd.concat(dask_results)
        else:
            preannotations = model.predict(m, pool, patch_size=patch_size, patch_overlap=patch_overlap, min_score=min_score)
            preannotations = pd.concat(preannotations)
        
        # Sort images by total number of predictions
        chosen_images = preannotations.groupby("image_path").size().sort_values(ascending=False).head(n).index.tolist()
        
        # Get full path
        chosen_images = [os.path.join(image_dir, image) for image in chosen_images]

    return chosen_images