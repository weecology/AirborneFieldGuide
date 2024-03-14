GitHub Copilot: # Deepforest Rapid Annotation and Model Iteration Pipeline

This repository contains a pipeline for rapid annotation and model iteration using the DeepForest model. The pipeline is designed to automate the process of annotating images, predicting labels, uploading images to an annotation platform, and training the model.

## Design

0. Install label-studio with a server url. Create a project with bounding box labels.
1. Get API key from label-studio account -> API, save as 
```
os.environ["LABEL_STUDIO_API_KEY"] = api_key
```
2. Create a config file, copy the sample ones in the repo.
3. Train an initial model or use random weights, save this model in the config["checkpoint_dir"]
4. Run a chron job for the pipeline to check label studio 

To create a cron job that runs a Python script once a day at 3pm Eastern Time, you can use the crontab command in Linux. Here's how you can do it:

Open the crontab file for editing by running crontab -e in your terminal.
Add the following line to the file:
```
0 15 * * * cd /path/to/script && /usr/bin/python3 drone_pipeline.py
```
This line tells cron to run the drone_pipeline.py script at 3pm (15:00) every day. Replace /path/to/script with the actual path to your drone_pipeline.py script.

## Code Overview

The main function in this pipeline is `iterate()`, which performs the following steps:

1. If a model checkpoint is provided, it loads the existing model and evaluates it. Otherwise, it skips the evaluation step.
2. It chooses images from the specified directory based on the evaluation results.
3. It uses the model to predict labels for the chosen images.
4. It uploads the predicted images to the specified folder on the annotation platform.
5. It checks if the annotation event is complete. If it is, it performs the following steps:
   - Downloads the labeled annotations.
   - Moves the annotated images out of the pool.
   - Trains the model using the new annotations.
   - Saves the trained model to a checkpoint file.


# To Do

- [ ] Active learning strategy
