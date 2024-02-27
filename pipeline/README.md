GitHub Copilot: # Deepforest Rapid Annotation and Model Iteration Pipeline

This repository contains a pipeline for rapid annotation and model iteration using the DeepForest model. The pipeline is designed to automate the process of annotating images, predicting labels, uploading images to an annotation platform, and training the model.

## Code Overview

The main function in this pipeline is `iterate()`, which takes the following arguments:

- `image_dir`: The path to a directory of images to annotate.
- `user`: The username for uploading images to the annotation platform.
- `host`: The host URL of the annotation platform.
- `folder_name`: The name of the folder to upload images to.
- `model_checkpoint` (optional): The path to the model checkpoint file. If provided, the existing model will be loaded from this file.

The `iterate()` function performs the following steps:

1. If a model checkpoint is provided, it loads the existing model and evaluates it. Otherwise, it skips the evaluation step.
2. It chooses images from the specified directory based on the evaluation results.
3. It uses the model to predict labels for the chosen images.
4. It uploads the predicted images to the specified folder on the annotation platform.
5. It checks if the annotation event is complete. If it is, it performs the following steps:
   - Downloads the labeled annotations.
   - Moves the annotated images out of the pool.
   - Trains the model using the new annotations.
   - Saves the trained model to a checkpoint file.

## Usage

To run the pipeline, execute the script with the appropriate arguments. For example:

```python
server_url = 'serenity.ifas.ufl.edu'
folder_name = '/pgsql/retrieverdash/everglades-label-studio/everglades-data/input'
model_checkpoint = "/blue/ewhite/everglades/Zooniverse//20220910_182547/species_model.pl"
iterate(image_dir="/blue/ewhite/everglades/label_studio/", model_checkpoint=model_checkpoint, user="ben", host=server_url, folder_name=folder_name)
```

This will run the pipeline on the images in the specified directory, using the specified model checkpoint, and upload the images to the specified folder on the annotation platform.

# To Do

- [ ] Hipergator to Serenity Authentication is currently ssh with a password, this will not work non-iteractively.
- [ ] Instructions to set up a cron-job
- [ ] DeepForest pull request for kwargs to pass to trainer https://github.com/weecology/DeepForest/issues/546
- [ ] How to assess if annotations are complete
- [ ] Formatting label-studio
- [ ] Active learning strategy
- [ ] Unit tests