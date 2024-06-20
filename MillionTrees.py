import os
import yaml
from src import pipeline
from src.cluster import start

# Read config
config = yaml.safe_load(open("MillionTrees_config.yml"))

# Set the Label studio API key as env variable
with open("/blue/ewhite/everglades/label_studio/label_studio_api_key.txt", "r") as file:
    api_key = file.read().strip()
os.environ["LABEL_STUDIO_API_KEY"] = api_key

#client = start(gpus=5, mem_size="20GB")
#pipeline.config_pipeline(config=config, dask_client=client)
pipeline.config_pipeline(config=config)