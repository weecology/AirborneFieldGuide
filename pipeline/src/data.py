import os
import random
import glob
import shutil
import json
import pandas as pd
from active_learning import active_learner

def choose_images(image_dir, evaluation=None):
    """Choose images to annotate.
    
    Args:
        pool (list): A list of image paths to choose from.
    
    Returns:
        list: A list of image paths to annotate.
    """
    pool = glob.glob(os.path.join(image_dir,"*")) # Get all images in the data directory
    
    if evaluation is None:
        chosen_images = random.sample(pool, 3)
    else:   
        active_learner(evaluation, pool, strategy="uncertainty")

    return chosen_images


def create_label_studio_json(image_url, predictions):
    """Create a JSON string for the Label Studio API.
    """
    # Prepare data dictionary
    data = {
        "image": image_url
    }

    # Prepare predictions list
    predictions_list = []
    for prediction in predictions:
        prediction_dict = {
            "result": prediction.result,
            "score": prediction.score,
            "cluster": prediction.cluster
        }
        predictions_list.append(prediction_dict)

    # Prepare final dictionary
    final_dict = {
        "data": data,
        "predictions": predictions_list
    }

    # Convert dictionary to JSON string
    json_string = json.dumps(final_dict)

    return json_string

# check_if_complete label studio images are done
def check_if_complete(annotations):
    """Check if any new images have been labeled.
    
    Returns:
        bool: True if new images have been labeled, False otherwise.
    """

    if annotations.shape[0] > 0:
        return True
    else:
        return False

def convert_json_to_dataframe(json_string):
    # Load JSON string into dictionary
    data_dict = json.loads(json_string)

    # Extract relevant information
    image_url = data_dict['data']['image']
    annotations = data_dict['annotations']
    predictions = data_dict['predictions']

    # Prepare list to hold all bounding box data
    bounding_boxes = []

    # Extract bounding box data from annotations
    for annotation in annotations:
        for result in annotation['result']:
            box = result['value']
            box['label'] = result['from_name']
            box['source'] = 'annotation'
            box['image_url'] = image_url
            bounding_boxes.append(box)

    # Extract bounding box data from predictions
    for prediction in predictions:
        for result in prediction['result']:
            box = result['value']
            box['label'] = result['from_name']
            box['source'] = 'prediction'
            box['image_url'] = image_url
            bounding_boxes.append(box)

    # Convert list of bounding boxes to DataFrame
    df = pd.DataFrame(bounding_boxes)

    return df

# Move images from images_to_annotation to images_annotated 
def move_images(annotations, src_dir, dst_dir):
    """Move images from the images_to_annotate folder to the images_annotated folder.
    
    Args:
        annotations (list): A list of annotations.
    
    Returns:
        None
    """
    images = annotations.image_path.unique()
    for image in images:
        src = os.path.join(src_dir, os.path.basename(image))
        dst = os.path.join(dst_dir, os.path.basename(image))
                           
        shutil.move(src, dst)

def gather_data(train_dir):
    train_csvs = glob.glob(os.path.join(train_dir,"*.csv"))
    df = []
    for x in train_csvs:
        df.append(pd.read_csv(x))
    df = pd.concat(df)
    df.drop_duplicates(inplace=True)
    
    return df

