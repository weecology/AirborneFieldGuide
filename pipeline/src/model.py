from deepforest import main
import os
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CometLogger

def evaluate(model, test_csv):
    """Evaluate a model on labeled images.
    
    Args:
        model (main.deepforest): A trained deepforest model.
        test_csv (str): The path to a CSV file containing annotations. Images are assumed to be in the same directory
    Returns:
        dict: A dictionary of evaluation metrics.
    """
    model.config["validation"]["csv_file"] = test_csv
    model.config["validation"]["root_dir"] = os.path.dirname(test_csv)
    results = model.trainer.validate(model)

    return results

def load(path):
    """Load a trained model from disk.
    
    Args:
        path (str): The path to a model checkpoint.
    
    Returns:
        main.deepforest: A trained deepforest model.
    """

    return main.deepforest.load_from_checkpoint(path)

def train(model, annotations, test_csv, checkpoint_dir):
    """Train a model on labeled images.
    Args:
        image_paths (list): A list of image paths.
    
    Returns:
        main.deepforest: A trained deepforest model.
    """
    comet_logger = CometLogger(project_name="everglades-species", workspace="weecology")
    comet_logger.experiment.log_table("train.csv", train)

    annotations.to_csv(os.path.join(checkpoint_dir,"train.csv"), index=False)
    model.config["train"]["csv_file"] = os.path.join(checkpoint_dir,"train.csv")
    model.config["validation"]["csv_file"] = test_csv
    model.config["validation"]["root_dir"] = os.path.dirname(test_csv) 
    
    checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_dir)
    trainer = model.create_trainer(callbacks=[checkpoint_callback])
    model.trainer.fit(model)

    return model