import os
import pandas as pd
import json
import numpy as np
import cv2
import dask.dataframe as dd

splits = {'train': 'data/train-*.parquet', 'test': 'data/test-00000-of-00001.parquet'}
df = dd.read_parquet("hf://datasets/restor/tcd/" + splits["test"])
ldf = df.compute()
ldf = ldf[~(ldf.coco_annotations=='[]')]

# Create a directory to save the images
output_dir = "/blue/ewhite/b.weinstein/MillionTrees/images_to_annotate/Restor"
os.makedirs(output_dir, exist_ok=True)

# Group the dataframe by biome_name column
grouped_df = ldf.groupby("biome_name")

# Iterate over each unique biome_name value
annotations = []
for biome_name, group in grouped_df:
    # Select 10 images from each group
    selected_images = group.head(10)
    
    # Iterate over the selected images
    for index, row in selected_images.iterrows():
        # Read the image from bytes in the image column
        image_bytes = row["image"]
        
        # Save the image to the output directory
        image_path = os.path.join(output_dir, image_bytes["path"])
        # Read the image from bytes in the image column
        image_bytes = np.frombuffer(image_bytes["bytes"], dtype=np.uint8)
        image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

        # Save the image as PNG to the output directory
        image_path = os.path.splitext(image_path)[0] + ".png"
        cv2.imwrite(image_path, image)
        
        # Create an empty dataframe to store the bounding boxes        
        bbox_df = pd.DataFrame(columns=["xmin", "xmax", "ymin", "ymax", "image_path"])
        coco_annotations = row["coco_annotations"]
        # Parse the coco_annotations JSON
        coco_annotations = json.loads(coco_annotations)

        # Iterate over each bounding box in coco_annotations
        for bbox in coco_annotations:
            # Extract the xmin, xmax, ymin, ymax values
            xmin = bbox["bbox"][0]
            xmax = bbox["bbox"][0] + bbox["bbox"][2]
            ymin = bbox["bbox"][1]
            ymax = bbox["bbox"][1] + bbox["bbox"][3]
            
            # Append the bounding box and image path to the dataframe
            bbox_df = pd.DataFrame({"xmin": [xmin], "xmax": [xmax], "ymin": [ymin], "ymax": [ymax], "image_path": [image_path], "label": [bbox["category_id"]]})
            annotations.append(bbox_df)
df = pd.concat(annotations)
# Replace 1 with 'Tree' and 2 with 'Canopy' in label column
df["label"] = df["label"].replace({1: "Canopy", 2: "Tree"})
df["score"] = 1.0
df.to_csv("/blue/ewhite/b.weinstein/MillionTrees/images_to_annotate/Restor/annotations.csv", index=False)
