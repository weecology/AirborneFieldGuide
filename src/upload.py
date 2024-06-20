import paramiko
import os
import datetime
import pandas as pd
from label_studio_sdk import Client
import glob
import shutil
from PIL import Image

def label_studio_bbox_format(local_image_dir, preannotations):
    """Create a JSON string for a single image the Label Studio API.
    """
    predictions = []
    original_width = Image.open(os.path.join(local_image_dir,os.path.basename(preannotations.image_path.unique()[0]))).size[0]
    original_height = Image.open(os.path.join(local_image_dir,os.path.basename(preannotations.image_path.unique()[0]))).size[1]

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

def gather_data(train_dir, labels=None):
    """Gather data from a directory of CSV files.
    Args:
        train_dir (str): The directory containing the CSV files.
        labels (list): A list of labels to filter by.
    
    Returns:
        pd.DataFrame: A DataFrame containing the data.
    """ 
    train_csvs = glob.glob(os.path.join(train_dir,"*.csv"))
    df = []
    for x in train_csvs:
        df.append(pd.read_csv(x))
    df = pd.concat(df)
    df.drop_duplicates(inplace=True)

    # Filter labels
    if labels:
        df = df[df["label"].isin(labels)]
    
    return df


def connect_to_label_studio(url, project_name):
    """Connect to the Label Studio server.
    Args:
        port (int, optional): The port of the Label Studio server. Defaults to 8080.
        host (str, optional): The host of the Label Studio server. Defaults to "localhost". 
    Returns:
        str: The URL of the Label Studio server.
    """
    ls = Client(url=url, api_key=os.environ["LABEL_STUDIO_API_KEY"])
    ls.check_connection()

    # Look up existing name
    projects = ls.list_projects()
    project = [x for x in projects if x.get_params()["title"] == project_name][0]

    return project

def create_client(user, host, key_filename):
    # Download annotations from Label Studio
    # SSH connection with a user prompt for password
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(host, username=user, key_filename=key_filename)
    sftp = ssh.open_sftp()

    return sftp

def delete_completed_tasks(label_studio_project):
    # Delete completed tasks
    tasks = label_studio_project.get_labeled_tasks()
    for task in tasks:
        label_studio_project.delete_task(task["id"])

def import_image_tasks(label_studio_project, image_names, local_image_dir, predictions=None):
    """
    Import image tasks into Label Studio project.

    Args:
        label_studio_project (LabelStudioProject): The Label Studio project to import tasks into.
        image_names (list): List of image names to import as tasks.
        local_image_dir (str): The local directory where the images are stored.
        predictions (list, optional): List of predictions for each image. Defaults to None.

    Returns:
        None
    """
    import os

    tasks = []
    for index, image_name in enumerate(image_names):
        data_dict = {'image': os.path.join("/data/local-files/?d=input/", os.path.basename(image_name))}
        if predictions:
            prediction = predictions[index]
            # Skip predictions if there are none
            if prediction.empty:
                result_dict = []
            else:
                result_dict = [label_studio_bbox_format(local_image_dir, prediction)]
            upload_dict = {"data": data_dict, "predictions": result_dict}
        tasks.append(upload_dict)
    label_studio_project.import_tasks(tasks)

def download_completed_tasks(label_studio_project, train_csv_folder):
    labeled_tasks = label_studio_project.get_labeled_tasks()
    if not labeled_tasks:
        print("No new annotations")
        return None
    else:
        images, labels = [], []
    for labeled_task in labeled_tasks:
        image_path = os.path.basename(labeled_task['data']['image'])
        images.append(image_path)
        label_json = labeled_task['annotations'][0]["result"]
        if len(label_json) == 0:
            result = {
                    "image_path": image_path,
                    "xmin": None,
                    "ymin": None,
                    "xmax": None,
                    "ymax": None,
                    "label": None,
                    "annotator":labeled_task["annotations"][0]["created_username"]
                }
            result = pd.DataFrame(result, index=[0])
        else:
            result = convert_json_to_dataframe(label_json, image_path)
            result["annotator"] = labeled_task["annotations"][0]["created_username"]
        labels.append(result)

    annotations =  pd.concat(labels) 
    print("There are {} new annotations".format(annotations.shape[0]))
    annotations = annotations[~(annotations.label=="Help me!")]
    annotations.loc[annotations.label=="Unidentified White","label"] = "Unknown White"

    # Save csv in dir with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    train_path = os.path.join(train_csv_folder, "train_{}.csv".format(timestamp))
    annotations.to_csv(train_path, index=False)

    return annotations

def upload_images(sftp_client, images, folder_name):
    """
    Uploads a list of images to a remote server using SFTP.

    Args:
        sftp_client (SFTPClient): An instance of the SFTPClient class representing the SFTP connection.
        images (list): A list of image file paths to be uploaded.
        folder_name (str): The name of the folder on the remote server where the images will be uploaded.

    Returns:
        None

    Raises:
        Any exceptions that may occur during the file transfer.

    """
    # SCP file transfer
    for image in images:
        sftp_client.put(image, os.path.join(folder_name,"input",os.path.basename(image)))
        print(f"Uploaded {image} successfully")

def remove_annotated_images_remote_server(sftp_client, annotations, folder_name):
    """Remove images that have been annotated on the Label Studio server."""
    # Delete images using SSH
    for image in annotations.image_path.unique():
        remote_path = os.path.join(folder_name, "input", os.path.basename(image))
        # Archive annotations using SSH
        archive_annotation_path = os.path.join(folder_name, "archive", os.path.basename(image))
        # sftp check if dir exists
        try:
            sftp_client.listdir(os.path.join(folder_name, "archive"))
        except FileNotFoundError:
            raise FileNotFoundError("The archive directory {} does not exist.".format(os.path.join(folder_name, "archive")))
        
        sftp_client.rename(remote_path, archive_annotation_path)
        print(f"Archived {image} successfully")

