import os
import yaml
from src import pipeline
from src.cluster import start
import argparse

# Read config
parser = argparse.ArgumentParser()
parser.add_argument("project_name", help="Name of the project")
parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use")
args = parser.parse_args()

config_file = args.project_name + "_config.yml"
config = yaml.safe_load(open(config_file))

# Set the Label studio API key as env variable
with open("/blue/ewhite/everglades/label_studio/label_studio_api_key.txt", "r") as file:
    api_key = file.read().strip()
os.environ["LABEL_STUDIO_API_KEY"] = api_key

if args.gpus > 1:
    client = start(gpus=args.gpus, mem_size="20GB")
    pipeline.config_pipeline(config=config, dask_client=client)
else:
    pipeline.config_pipeline(config=config)
