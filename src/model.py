from logging import warn
from deepforest import main
import os
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CometLogger
import tempfile
import warnings
import glob

def evaluate(model, test_csv):
    """Evaluate a model on labeled images.
    
    Args:
        model (main.deepforest): A trained deepforest model.
        test_csv (str): The path to a CSV file containing annotations. Images are assumed to be in the same directory
    Returns:
        dict: A dictionary of evaluation metrics.
    """
    # create trainer
    model.create_trainer()
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
    # Check classes
    if path == "tree":
        snapshot = main.deepforest()
        snapshot.use_release()
    elif path == "bird":
        snapshot = main.deepforest()
        snapshot.use_bird_release()
    else:   
        snapshot = main.deepforest.load_from_checkpoint(path)

    return snapshot

def extract_backbone(path, annotations):
    snapshot = main.deepforest.load_from_checkpoint(path)
    warnings.warn("The number of classes in the model does not match the number of classes in the annotations. The backbone will be extracted and retrained.")
    label_dict = {value: index for index, value in enumerate(annotations.label.unique())}
    m = main.deepforest(num_classes=len(annotations.label.unique()), label_dict=label_dict, config_file="deepforest_config.yml")
    m.model.backbone.load_state_dict(snapshot.model.backbone.state_dict())
    m.model.head.regression_head.load_state_dict(snapshot.model.head.regression_head.state_dict())

    return m

def train(model, annotations, train_image_dir, checkpoint_dir, comet_project=None, comet_workspace=None):
    """Train a model on labeled images.
    Args:
        image_paths (list): A list of image paths.
        annotations (pd.DataFrame): A DataFrame containing annotations.
        train_image_dir (str): The directory containing the training images.
        checkpoint_dir (str): The directory to save model checkpoints.
        comet_project (str): The comet project name for logging. Defaults to None.
    
    Returns:
        main.deepforest: A trained deepforest model.
    """
    tmpdir = tempfile.gettempdir()

    annotations.to_csv(os.path.join(tmpdir,"train.csv"), index=False)
    model.config["train"]["csv_file"] = os.path.join(tmpdir,"train.csv")
    model.config["train"]["root_dir"] = train_image_dir
    
    if comet_project:
        comet_logger = CometLogger(project_name=comet_project, workspace=comet_workspace)
        comet_logger.experiment.log_parameters(model.config)
        comet_logger.experiment.log_table("train.csv", annotations)
        #checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_dir)
        model.create_trainer(logger=comet_logger)
    else:
        model.create_trainer()
    
    model.trainer.fit(model)

    return model

def get_latest_checkpoint(checkpoint_dir, annotations):
    #Get model with latest checkpoint dir, if none exist make a new model
    if os.path.exists(checkpoint_dir):
        checkpoints = glob.glob(os.path.join(checkpoint_dir,"*.ckpt"))
        if len(checkpoints) > 0:
            checkpoints.sort()
            checkpoint = checkpoints[-1]
            m = load(checkpoint)
        else:
            warn("No checkpoints found in {}".format(checkpoint_dir))
            label_dict = {value: index for index, value in enumerate(annotations.label.unique())}
            m = main.deepforest(config_file="Airplane/deepforest_config.yml", label_dict=label_dict)
    else:
        os.makedirs(checkpoint_dir)
        m = main.deepforest(config_file="Airplane/deepforest_config.yml")
    
    return m

def predict(image_paths, patch_size, patch_overlap, min_score, m=None, model_path=None):
    """Predict bounding boxes for images
    Args:
        m (main.deepforest): A trained deepforest model.
        image_paths (list): A list of image paths.  
    Returns:
        list: A list of image predictions.
    """
    if model_path:
        m = load(model_path)
    else:
        if m is None:
            raise ValueError("A model or model_path is required for prediction.")
    
    # if no trainer, create one
    if m.trainer is None:
        m.create_trainer()

    predictions = []
    
    for image_path in image_paths:
        prediction = m.predict_tile(raster_path=image_path, return_plot=False, patch_size=patch_size, patch_overlap=patch_overlap)
        prediction = prediction[prediction.score > min_score]
        predictions.append(prediction)

    return predictions