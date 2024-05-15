"""
Upload existing annotations and images to the label studio instance. In this use case, we have images and provisional annotations that we can review
"""
import pandas as pd
from src import upload
import yaml
import os

# Read config
# Set the Label studio API key as env variable
with open("/blue/ewhite/everglades/label_studio/label_studio_api_key.txt", "r") as file:
    api_key = file.read().strip()
os.environ["LABEL_STUDIO_API_KEY"] = api_key

config = yaml.safe_load(open("pipeline/airplane_config.yml"))
csv_file = "/blue/ewhite/everglades/Airplane/annotations/converted_photoshop/tiles/split_annotations.csv"
sftp_client = upload.create_client(user=config["user"], host=config["server_url"], key_filename=config["key_filename"])
label_studio_project = upload.connect_to_label_studio(url=config["label-studio-url"], project_name=config["label-studio-project"])

# Load the csv file
df = pd.read_csv(csv_file)

images = df["image_path"].unique()
# Full path
images = [os.path.join(os.path.dirname(csv_file), image) for image in images]
upload.upload_images(sftp_client=sftp_client, images=images, folder_name=config["folder_name"])

# Add a dummy score 
df["score"] = 1

predictions = []
for image_name in images:
    predictions.append(df[df["image_path"] == os.path.basename(image_name)])
upload.import_image_tasks(label_studio_project=label_studio_project, 
                          image_names=images, 
                          local_image_dir=os.path.dirname(csv_file), 
                          predictions=predictions)
