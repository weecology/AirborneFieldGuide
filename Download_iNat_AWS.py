import requests
import gzip
import csv
import os
import dask.dataframe as dd
from distributed import Client
import pandas as pd

client = Client(processes=False)

# Step 0, download the CSV file and gzip
# aws s3 cp s3://inaturalist-open-data/photos.csv.gz /blue/ewhite/b.weinstein/BOEM/iNat
# gzip -d photos.csv.gz

# Example usage
local_csv_path = '/blue/ewhite/b.weinstein/BOEM/iNat/photos.csv'

def download_images(species, num_images, output_dir,local_csv_path):
    client = Client(processes=False)

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create a subdirectory for the species if it doesn't exist
    species_dir = os.path.join(output_dir, species)
    if not os.path.exists(species_dir):
        os.makedirs(species_dir)

    df = dd.read_csv(local_csv_path)

    # Filter the dataframe for the specified species
    filtered_df = df[df['species'] == species].head(num_images, compute=True)

    count = 0
    for _, row in filtered_df.iterrows():
        if count >= num_images:
            break
        photo_id = row['photo_id']
        extension = row['extension']
        image_url = f'https://inaturalist-open-data.s3.amazonaws.com/photos/{photo_id}/original.{extension}'
        image_response = requests.get(image_url)
        if image_response.status_code == 200:
            with open(os.path.join(species_dir, f'{photo_id}.{extension}'), 'wb') as img_file:
                img_file.write(image_response.content)
            count += 1

# Example usage for Parasitic Jaeger (Stercorarius parasiticus)
download_images('Stercorarius parasiticus', 5)

# read in a csv, loop through species and download images
species_df = pd.read_csv(local_csv_path)
species_list = species_df['species'].unique()
for species in species_list:
    download_images(species, 5)
