import os
import subprocess
import boto3

# Set the S3 bucket URL
bucket_url = "s3://twi-aviandata/avian_monitoring/high_resolution_photos/"

# Create a Boto3 S3 client
s3_client = boto3.client('s3')

# List all subfolders in the S3 bucket
subfolders = subprocess.check_output(['aws', 's3', 'ls', bucket_url]).decode('utf-8').split('\n')
subfolders = [folder.split(' ')[-2] for folder in subfolders if folder]

# Download two images from each subfolder
for subfolder in subfolders:
    # List all objects in the subfolder
    objects = s3_client.list_objects_v2(Bucket='twi-aviandata', Prefix=f"avian_monitoring/high_resolution_photos/{subfolder}/")['Contents']
    
    # Download the first two images
    for i in range(2):
        object_key = objects[i]['Key']
        file_name = os.path.basename(object_key)
        s3_client.download_file('twi-aviandata', object_key, file_name)
        print(f"Downloaded {file_name} from {subfolder}")