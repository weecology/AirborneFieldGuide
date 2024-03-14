import os
import glob
import shutil
import pandas as pd
import PIL

def label_studio_bbox_format(local_image_dir, preannotations):
    """Create a JSON string for a single image the Label Studio API.
    """
    predictions = []
    original_width = PIL.Image.open(os.path.join(local_image_dir,os.path.basename(preannotations.image_path.unique()[0]))).size[0]
    original_height = PIL.Image.open(os.path.join(local_image_dir,os.path.basename(preannotations.image_path.unique()[0]))).size[1]

    for index, row in preannotations.iterrows():
        result = {
            "value":{
                "x": row['xmin']/original_width*100,
                "y": row['ymin']/original_height*100,
                "width": (row['xmax'] - row['xmin'])/original_width*100,
                "height": (row['ymax'] - row['ymin'])/original_height*100,
                "rotation": 0,
                "rectanglelabels": [row["label"]]
            },
            "score": row["score"],
            "to_name": "image",
            "type": "rectanglelabels",
            "from_name": "label",
            "original_width": original_width,
            "original_height": original_height
        }
        predictions.append(result)
    # As a dict
    return {"result": predictions}

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

# con for a json that looks like 
#{'id': 539, 'created_username': ' vonsteiny@gmail.com, 10', 'created_ago': '0\xa0minutes', 'task': {'id': 1962, 'data': {...}, 'meta': {}, 'created_at': '2023-01-18T20:58:48.250374Z', 'updated_at': '2023-01-18T20:58:48.250387Z', 'is_labeled': True, 'overlap': 1, 'inner_id': 381, 'total_annotations': 1, ...}, 'completed_by': {'id': 10, 'first_name': '', 'last_name': '', 'email': 'vonsteiny@gmail.com'}, 'result': [], 'was_cancelled': False, 'ground_truth': False, 'created_at': '2023-01-30T21:43:35.447447Z', 'updated_at': '2023-01-30T21:43:35.447460Z', 'lead_time': 29.346, 'parent_prediction': None, 'parent_annotation': None}
    
def convert_json_to_dataframe(x, image_path):
    # Loop through annotations and convert to pandas {'original_width': 6016, 'original_height': 4008, 'image_rotation': 0, 'value': {'x': 94.96474718276704, 'y': 22.132321974413898, 'width': 1.7739074476466308, 'height': 2.2484415320942235, 'rotation': 0, 'rectanglelabels': [...]}, 'id': 'UeovfQERjL', 'from_name': 'label', 'to_name': 'image', 'type': 'rectanglelabels', 'origin': 'manual'}
    results = []
    for annotation in x:
        xmin = annotation["value"]["x"]/100 * annotation["original_width"]
        ymin = annotation["value"]["y"]/100 * annotation["original_height"]
        xmax = (annotation["value"]["width"]/100 + annotation["value"]["x"]/100 ) * annotation["original_width"]
        ymax = (annotation["value"]["height"]/100 + annotation["value"]["y"]/100) * annotation["original_height"]
        label = annotation["value"]["rectanglelabels"][0]

        # Create dictionary
        result = {
            "image_path": image_path,
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax,
            "label": label,
        }

        # Append to list
        results.append(result)
        df = pd.DataFrame(results)
        
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
                           
        try:
            shutil.move(src, dst)
        except FileNotFoundError:
            continue

def gather_data(train_dir):
    train_csvs = glob.glob(os.path.join(train_dir,"*.csv"))
    df = []
    for x in train_csvs:
        df.append(pd.read_csv(x))
    df = pd.concat(df)
    df.drop_duplicates(inplace=True)
    
    return df

