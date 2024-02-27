from src import data, model, predict, upload
import datetime
import os
import configparser

def iterate(checkpoint_dir, images_to_annotate_dir, annotated_image_dir, test_csv, user, host, folder_name, model_checkpoint=None):
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

    Returns:
        None
    """
    # Check event for there new annotations
    # Download labeled annotations
    annotations = data.download_annotations()
    complete = data.check_if_complete(annotations)
    
    if complete:
        # Save new training data with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        train_path = os.path.join(annotated_image_dir, "train_{}.csv".format(timestamp))
        annotations.to_csv(train_path, index=False)

        # Load existing model
        if model_checkpoint:
            m = model.load(model_checkpoint)
            evaluation = model.evaluate(m)
            print(evaluation)
        elif os.path.exists(checkpoint_dir):
            model.get_latest_checkpoint(checkpoint_dir)
        else:
            evaluation = None

        # Move annotated images out of local pool
        data.move_images(src_dir=images_to_annotate_dir,dst_dir=annotated_image_dir, annotations=annotations)

        # Remove images that have been labeled on the label_studio server
        #upload.remove_annotated_image_remote_server(annotations)

        # Train model and save checkpoint
        train_df = data.gather_training_data()
        model = model.train(train_df, test_csv, checkpoint_dir)

        # Choose new images to annotate
        images = data.choose_images(images_to_annotate_dir, evaluation)

        # Predict images
        preannotations = predict.predict(model, images)

        # Upload images to annotation platform
        upload.upload_images(images, user, host, folder_name)
        upload.upload_preannotations(preannotations, user, host, folder_name)

if __name__ == "__main__":

    # Read config
    config = configparser.ConfigParser()
    
    iterate(
        checkpoint_dir="/blue/ewhite/everglades/label_studio/checkpoints",
        images_to_annotate_dir=config["images_to_annotate_dir"],
        annotated_images_dir=config["annotated_images_dir"],
        model_checkpoint=config["model_checkpoint"],
        user=config["ben"],
        host=config["server_url"],
        folder_name=config["folder_name"]
        )

