import paramiko
import os
import datetime
from src import data
import pandas as pd
from label_studio_sdk import Client

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

def import_image_tasks(label_studio_project,image_names, local_image_dir, predictions=None):
    # Get project
    tasks = []
    for index, image_name in enumerate(image_names):
        data_dict = {'image': os.path.join("/data/local-files/?d=input/",os.path.basename(image_name))}
        if predictions:
            prediction = predictions[index]
            #Skip predictions if there are none
            if prediction.empty:
                result_dict = []
            else:
                result_dict = [data.label_studio_bbox_format(local_image_dir, prediction)]
            upload_dict = {"data":data_dict, "predictions":result_dict}
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
            result = data.convert_json_to_dataframe(label_json, image_path)
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

