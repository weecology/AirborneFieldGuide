import os
import subprocess

def download_images_from_s3(url, num_images):
    # Extract bucket name and prefix from the URL
    bucket_name = url.split('/')[2]
    prefix = '/'.join(url.split('/')[3:])

    # List all objects in the specified bucket and prefix
    command = f"aws s3 ls s3://{bucket_name}/{prefix} --recursive"
    output = subprocess.check_output(command, shell=True).decode('utf-8')

    # Create the target directory if it doesn't exist
    target_directory = "/orange/ewhite/b.weinstein/DeepWaterHorizon/"
    os.makedirs(target_directory, exist_ok=True)

    # Iterate over the objects and download two images from each folder
    count = 0
    for line in output.splitlines():
        if count < num_images:
            # Extract the object key from the output line
            object_key = line.split()[-1]

            # Download the image using wget and save it in the target directory
            command = f"wget s3://{bucket_name}/{object_key} -P {target_directory}"
            subprocess.run(command, shell=True)

            count += 1

download_images_from_s3("s3://deepwaterhorizon", 2)