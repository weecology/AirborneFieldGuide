from logging import warn
from deepforest import main, preprocess
import os
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CometLogger
import tempfile
import warnings
import glob
import pandas as pd
import dask.array as da
from deepforest import visualize

def evaluate(model, test_csv, image_root_dir):
    """Evaluate a model on labeled images.
    
    Args:
        model (main.deepforest): A trained deepforest model.
        test_csv (str): The path to a CSV file containing annotations.
        image_root_dir (str): The directory containing the test images.
    Returns:
        dict: A dictionary of evaluation metrics.
    """
    # create trainer
    model.create_trainer()
    model.config["validation"]["csv_file"] = test_csv
    model.config["validation"]["root_dir"] = image_root_dir
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
    new_labels = annotations.label.unique()
    new_labels = new_labels[~pd.isnull(new_labels)]

    label_dict = {value: index for index, value in enumerate(new_labels)}
    m = main.deepforest(num_classes=len(), label_dict=label_dict, config_file="deepforest_config.yml")
    m.model.backbone.load_state_dict(snapshot.model.backbone.state_dict())
    m.model.head.regression_head.load_state_dict(snapshot.model.head.regression_head.state_dict())

    return m

def preprocess_images(annotations, root_dir, save_dir, limit_empty_frac=0.5, patch_size=450, patch_overlap=0):
    """Cut images into GPU friendly chunks"""
    crop_annotations = []
    for image_path in annotations.image_path.unique():
        # check if crops with that prefix exist in save_dir
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        crop_csv = "{}.csv".format(os.path.join(save_dir, image_name))
        if os.path.exists(crop_csv):
            warn("Crops for {} already exist in {}. Skipping.".format(crop_csv, save_dir))
            crop_annotations.append(pd.read_csv(crop_csv))
            continue
        full_path = os.path.join(root_dir, image_path)
        crop_annotation = preprocess.split_raster(
            path_to_raster=full_path,
            annotations_file=annotations,
            patch_size=patch_size,
            patch_overlap=patch_overlap,
            save_dir=save_dir,
            root_dir=root_dir,
            allow_empty=True
        )
        crop_annotation_empty = crop_annotation[crop_annotation.xmin==0]
        crop_annotation_non_empty = crop_annotation[crop_annotation.xmin!=0]
        crop_annotation_empty = crop_annotation_empty.sample(frac=limit_empty_frac)
        crop_annotation = pd.concat([crop_annotation_empty, crop_annotation_non_empty])
        crop_annotations.append(crop_annotation)
    
    crop_annotations = pd.concat(crop_annotations)
    return crop_annotations
  
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
        #plot_names = visualize.plot_prediction_dataframe(df=non_empty.head(5), root_dir=train_image_dir, savedir=tmpdir)
        #for plot_name in plot_names:
        comet_logger.experiment.log_image(os.path.join(tmpdir,plot_name))
        comet_logger.experiment.log_parameters(model.config)
        comet_logger.experiment.log_table("train.csv", annotations)
        checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_dir)
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

def _predict_list_(image_paths, min_score, patch_size, patch_overlap, model_path, m=None):
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

def predict(image_paths, patch_size, patch_overlap, min_score, m=None, model_path=None,dask_client=None):
    """Predict bounding boxes for images
    Args:
        m (main.deepforest): A trained deepforest model.
        image_paths (list): A list of image paths.  
    Returns:
        list: A list of image predictions.
    """

    if dask_client:
        # load model on each client
        def update_sys_path():
            import sys
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        dask_client.run(update_sys_path)

        # Load model on each client
        dask_pool = da.from_array(image_paths, chunks=len(image_paths)//len(dask_client.ncores()))
        blocks = dask_pool.to_delayed().ravel()
        block_futures = []
        for block in blocks:
            block_future = dask_client.submit(_predict_list_,
                                              image_paths=block.compute(),
                                              patch_size=patch_size,
                                              patch_overlap=patch_overlap,
                                              min_score=min_score,
                                              model_path=model_path,
                                              m=m)
            block_futures.append(block_future)
        # Get results
        predictions = []
        for block_result in block_futures:
            block_result = block_result.result()
            predictions.append(pd.concat(block_result))
    else:
        predictions = _predict_list_(image_paths=image_paths, patch_size=patch_size, patch_overlap=patch_overlap, min_score=min_score, model_path=model_path, m=m)

    return predictions