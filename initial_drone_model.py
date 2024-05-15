# Train an initial deepforest model for the drone pipeline
from src import model
import yaml
import pandas as pd

# read config
config = yaml.safe_load(open("Airplane/airplane_config.yml"))

# Get weights from latest drone model
# Train data
annotations = pd.read_csv("")

# Test data
m = model.extract_backbone('/blue/ewhite/everglades/Zooniverse/20220910_182547/species_model.pl', annotations)
model.train(
    model=m,
    annotations=annotations,
    test_csv=config["test_csv"],
    checkpoint_dir=config["checkpoint_dir"],
    train_image_dir=config["annotated_images_dir"])