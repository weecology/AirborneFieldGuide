import pandas as pd
from src import data, model, upload, active_learning
from deepforest import visualize
import os

def config_pipeline(config):
    iterate(
        checkpoint_dir=config["checkpoint_dir"],
        images_to_annotate_dir=config["images_to_annotate_dir"],
        annotated_images_dir=config["annotated_images_dir"],
        model_checkpoint=config["model_checkpoint"],
        user=config["user"],
        host=config["server_url"],
        test_csv=config["test_csv"],
        train_csv_folder=config["train_csv_folder"],
        folder_name=config["folder_name"],
        key_filename=config["key_filename"],
        annotation_csv=config["annotation_csv"],
        patch_size=config["patch_size"],
        patch_overlap=config["patch_overlap"],
        label_studio_project_name=config["label-studio-project"],
        label_studio_url=config["label-studio-url"],
        force_run=config["force_run"],
        skip_train=config["skip_train"],
        min_score=config["min_score"]
    )

def iterate(
        checkpoint_dir,
        images_to_annotate_dir,
        annotated_images_dir,
        test_csv,
        user,
        host,
        folder_name,
        key_filename,
        patch_size,
        patch_overlap,
        label_studio_url,
        label_studio_project_name,
        train_csv_folder,
        min_score=0.3,
        model_checkpoint=None,
        annotation_csv=None,
        force_run=False,
        skip_train=False):
    """A Deepforest pipeline for rapid annotation and model iteration.

    Args:
        checkpoint_dir: The path to a directory for saving model checkpoints.
        images_to_annotate_dir: The path to a directory of images to annotate.
        annotated_images_dir: The path to a directory of annotated images.
        test_csv: The path to a CSV file containing annotations. Images are assumed to be in the same directory.
        user (str): The username for uploading images to the annotation platform.
        host (str): The host URL of the annotation platform.
        folder_name (str): The name of the folder to upload images to.
        model_checkpoint (str, optional): The path to the model checkpoint file. Defaults to None.
        annotation_csv (str, optional): The path to the CSV file containing annotations. Defaults to None. This will skip checking the server for debugging
        patch_size: The size of the image patches to predict on for main.deepforest.predict_tile.
        patch_overlap: The amount of overlap between image patches.
        label_studio_url: The URL of the Label Studio server.
        label_studio_project_name: The name of the Label Studio project.
        train_csv_folder: The path to a directory of CSV files containing annotations.
        min_score: The minimum score for a prediction to be included in the annotation platform.
        force_run: If True, will run the pipeline even if there are no new annotations. Defaults to False.
        skip_train: If True, will skip training the model. Defaults to False.
    Returns:
        None
    """
    # Check event for there new annotations
    # Download labeled annotations
    sftp_client = upload.create_client(user=user, host=host, key_filename=key_filename)
    label_studio_project = upload.connect_to_label_studio(url=label_studio_url, project_name=label_studio_project_name)

    if force_run:
        annotations = None
        complete = True
    elif annotation_csv is None:
        annotations = upload.download_completed_tasks(label_studio_project=label_studio_project, train_csv_folder=train_csv_folder)
        if annotations is None:
            print("No new annotations")
            complete = False
        else:
            complete = True
    else:
        annotations = pd.read_csv(annotation_csv)
        complete = True
    if complete:
        # Load existing model
        if model_checkpoint:
            m = model.load(model_checkpoint)
        elif os.path.exists(checkpoint_dir):
            m = model.get_latest_checkpoint(checkpoint_dir, annotations)
        else:
            evaluation = None

        # Train model and save checkpoint
        if not skip_train:
            # Choose new images to annotate
            m.config["validation"]["csv_file"] = test_csv
            m.config["validation"]["root_dir"] = os.path.dirname(test_csv) 
            before_evaluation = model.evaluate(m, test_csv=test_csv)
            print(before_evaluation)

            train_df = data.gather_data(annotated_images_dir)

            # View test images overlaps, just a couple debugs
            visualize.plot_prediction_dataframe(df=pd.read_csv(test_csv).head(100), root_dir=os.path.dirname(test_csv), savedir="/blue/ewhite/everglades/label_studio/test_plots")
            visualize.plot_prediction_dataframe(df=train_df.head(100), root_dir=annotated_images_dir, savedir="/blue/ewhite/everglades/label_studio/test_plots")

            m = model.train(model=m, annotations=train_df, test_csv=test_csv, checkpoint_dir=checkpoint_dir, train_image_dir=annotated_images_dir)

            # Choose new images to annotate
            evaluation = model.evaluate(m, test_csv=test_csv)
            print(evaluation)

        # Move annotated images out of local pool
        if annotations is not None:
            data.move_images(src_dir=images_to_annotate_dir, dst_dir=annotated_images_dir, annotations=annotations)

        # Choose local images to annotate
        images = active_learning.choose_images(image_dir=images_to_annotate_dir, evaluation=None, strategy="random", n=10)

        # Predict images
        preannotations = model.predict(m, images, patch_size=patch_size, patch_overlap=patch_overlap, min_score=min_score)
        
        # Upload images to annotation platform
        upload.upload_images(sftp_client=sftp_client, images=images, folder_name=folder_name)
        upload.import_image_tasks(label_studio_project=label_studio_project, image_names=images, local_image_dir=images_to_annotate_dir, predictions=preannotations)

        # Delete completed tasks
        upload.delete_completed_tasks(label_studio_project=label_studio_project)