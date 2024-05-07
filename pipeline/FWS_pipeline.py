import os
import yaml
from src import pipeline
from dask_jobqueue import SLURMCluster

# Read config
config = yaml.safe_load(open("FWS_config.yml"))

# Set the Label studio API key as env variable
with open("/blue/ewhite/everglades/label_studio/label_studio_api_key.txt", "r") as file:
    api_key = file.read().strip()
os.environ["LABEL_STUDIO_API_KEY"] = api_key

# Create a dask client from dask-jobqueue

extra_args = [
    "--error=/orange/idtrees-collab/logs/dask-worker-%j.err", "--account=ewhite",
    "--output=/orange/idtrees-collab/logs/dask-worker-%j.out", "--partition=gpu",
    "--gpus=1"
]

cluster = SLURMCluster(processes=1,
                        cores=3,
                        memory="20GB",
                        walltime='24:00:00',
                        job_extra=extra_args,
                        extra=['--resources gpu=1'],
                        nanny=False,
                        scheduler_options={"dashboard_address": ":8787"},
                        local_directory="/orange/idtrees-collab/tmp/",
                        death_timeout=300)
print(cluster.job_script())
# How many gpus to scale to
cluster.scale(8)

pipeline.config_pipeline(config, dask_client=cluster)