import requests
import gzip
import csv
import os
import dask.dataframe as dd
from distributed import Client

client = Client(processes=False)

def download_and_load_csv(url, local_path):
    # Download the CSV file if it doesn't already exist
    if not os.path.exists(local_path):
        response = requests.get(url)
        with open(local_path, 'wb') as f:
            f.write(response.content)

    # Unzip the CSV file if the unzipped version doesn't already exist
    unzipped_path = local_path[:-3]
    if not os.path.exists(unzipped_path):
        with gzip.open(local_path, 'rb') as f_in:
            with open(unzipped_path, 'wb') as f_out:
                f_out.write(f_in.read())

    # Load the CSV file with Dask
    df = dd.read_csv(unzipped_path)
    return df

# Example usage
csv_url = 'https://inaturalist-open-data.s3.amazonaws.com/photos.csv.gz'
local_csv_path = 'photos.csv.gz'
df = download_and_load_csv(csv_url, local_csv_path)

def download_images(species, num_images, output_dir):
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create a subdirectory for the species if it doesn't exist
    species_dir = os.path.join(output_dir, species)
    if not os.path.exists(species_dir):
        os.makedirs(species_dir)

    # URL for the photos.csv.gz file
    csv_url = 'https://inaturalist-open-data.s3.amazonaws.com/photos.csv.gz'
    # Load the CSV file with Dask
    df = download_and_load_csv(csv_url, local_csv_path)

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
