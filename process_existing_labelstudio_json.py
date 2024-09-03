# Process existing label-studio json objects
from src.upload import convert_json_to_dataframe, gather_data
import glob
import datetime
import os
import pandas as pd
import json

train_csv_folder = "/blue/ewhite/everglades/label_studio/airplane/annotated_images/csvs/"
label_jsons = glob.glob("/blue/ewhite/everglades/label_studio/airplane/annotated_images/csvs/label_studio/*.json")
labels = []
for label_json in label_jsons:
    with open(label_json, 'r') as f:
        json_data = json.load(f)
        annotation = json_data["annotations"][0]["result"]
        result = convert_json_to_dataframe(annotation)
        image_path = os.path.basename(json_data['data']['image'])
        result["image_path"] = image_path
    labels.append(result)

annotations =  pd.concat(labels) 
annotations = annotations[~(annotations.label=="Help me!")]
annotations.loc[annotations.label=="Unidentified White","label"] = "Unknown White"

# Save csv in dir with timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
train_path = os.path.join(train_csv_folder, "train_{}.csv".format(timestamp))

#drop duplicates
annotations = annotations.drop_duplicates()
annotations.to_csv(train_path, index=False)
print("There are {} new annotations".format(annotations.shape[0]))

merged = gather_data(train_csv_folder)
