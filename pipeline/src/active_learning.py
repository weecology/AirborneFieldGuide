import random
import os
import glob

def choose_images(evaluation, image_dir, strategy, n=10):
    """Choose images to annotate.
    Args:
        evaluation (dict): A dictionary of evaluation metrics.
        image_dir (str): The path to a directory of images.
        strategy (str): The strategy for choosing images.
        n (int, optional): The number of images to choose. Defaults to 10.
    Returns:
        list: A list of image paths.
    """
    pool = glob.glob(os.path.join(image_dir,"*")) # Get all images in the data directory
    if strategy=="random":
        chosen_images = random.sample(pool, n)
        return chosen_images
    else:
        raise ValueError("The strategy {} is not implemented.".format(strategy))
    
    return chosen_images