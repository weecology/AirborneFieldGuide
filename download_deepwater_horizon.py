import os
import subprocess
import boto3
import pandas as pd


# Set the S3 bucket URL
bucket_url = "s3://twi-aviandata/avian_monitoring/high_resolution_photos/"

# Create a Boto3 S3 client
s3_client = boto3.client('s3')

# List all subfolders in the S3 bucket
subfolders = subprocess.check_output(['aws', 's3', 'ls', bucket_url]).decode('utf-8').split('\n')
subfolders = [folder.split(' ')[-1] for folder in subfolders if folder]

objects = s3_client.list_objects_v2(Bucket='twi-aviandata', Prefix=f"avian_monitoring/high_resolution_photos/")
# get filenames of all objects

# Get filenames of all objects
filenames = [obj['Key'] for obj in objects['Contents']]

#get full folder path for each object
folder_paths = [os.path.dirname(filename) for filename in filenames]

# get unique paths and select two examples of each using a pandas dataframe
df = pd.DataFrame({'folder_path': folder_paths,"filename":filenames})
# Sample two images for each unique folder_path
sample_images = df.groupby('folder_path').apply(lambda x: x.sample(2,replace=True)).reset_index(drop=True)
files_to_download = sample_images.filename.unique()
len(files_to_download)

for file in files_to_download:
    # Download the file
    filename_to_save = file.replace("/", "_") 
    # no spaces
    filename_to_save = filename_to_save.replace(" ", "_")  
    #  
    s3_client.download_file('twi-aviandata', file, filename_to_save)
    print(f"Downloaded {file}")

