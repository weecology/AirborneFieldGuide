import os
import dask.dataframe as dd

splits = {'train': 'data/train-*.parquet', 'test': 'data/test-00000-of-00001.parquet'}
df = dd.read_parquet("hf://datasets/restor/tcd/" + splits["test"])
ldf = df.compute()

# Create a directory to save the images
output_dir = "/blue/ewhite/b.weinstein/MillionTrees/images_to_annotate/Restor"
os.makedirs(output_dir, exist_ok=True)

# Group the dataframe by biome_name column
grouped_df = ldf.groupby("biome_name")

# Iterate over each unique biome_name value
for biome_name, group in grouped_df:
    # Select 10 images from each group
    selected_images = group.head(10)
    
    # Iterate over the selected images
    for index, row in selected_images.iterrows():
        # Read the image from bytes in the image column
        image_bytes = row["image"]
        
        # Save the image to the output directory
        image_path = os.path.join(output_dir, image_bytes["path"])
        with open(image_path, "wb") as f:
            f.write(image_bytes["bytes"])