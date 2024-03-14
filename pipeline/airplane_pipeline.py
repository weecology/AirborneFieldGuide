import os
import yaml
from src import pipeline

# Read config
config = yaml.safe_load(open("Airplane/airplane_config.yml"))

# Set the Label studio API key as env variable
with open("/blue/ewhite/everglades/label_studio/label_studio_api_key.txt", "r") as file:
    api_key = file.read().strip()
os.environ["LABEL_STUDIO_API_KEY"] = api_key

pipeline.config_pipeline(config)
