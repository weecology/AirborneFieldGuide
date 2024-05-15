# Airborne Field Guide

# Installation

```
conda create --name AirborneFieldGuide
conda activate AirborneFieldGuide
pip install -r requirements.txt
```

# What does this repo do?

This repo is a development space for airborne machine learning model development using active learning and label-studio for rapid iteration. Unlabeled data is uploaded to label studio alongside existing model predictions. Human annotators correct and update these 'pre-annotations'. The data is then downloaded nightly to train a new model. 

## Contents

The pipeline contains two parts

* A config .yml file that specifies task specific details such as label-studio project and active learning strategy
* A <name>_pipeline.py file that loads the config and runs the iteration. There is also an optional submit_<name>.sh script for deploying on SLURM clusters.

### Config

A general example of the yaml
```
# Remote server for image hosting
user: 'ben'
server_url: 'serenity.ifas.ufl.edu'
key_filename: '/home/b.weinstein/.ssh/id_rsa.pub'
folder_name: '/pgsql/retrieverdash/everglades-label-studio/everglades-data'

# Model Training
# Model checkpoints are saved in the checkpoint_dir, model_checkpoint is the path to the model checkpoint and is honored first
model_checkpoint: '/blue/ewhite/everglades/Zooniverse//20230426_082517/species_model.pl'
test_csv: '/blue/ewhite/everglades/Zooniverse/cleaned_test/test_resized_no_nan.csv'
images_to_annotate_dir: '/blue/ewhite/everglades/label_studio/airplane/images_to_annotate'
checkpoint_dir: '/blue/ewhite/everglades/Zooniverse/'
train_csv_folder: /blue/ewhite/everglades/label_studio/airplane/annotated_images/csvs

# Active learning
annotated_images_dir: '/blue/ewhite/everglades/label_studio/airplane/annotated_images'
#annotation_csv: /blue/ewhite/everglades/label_studio/airplane/annotated_images/train_20231214_093907.csv
annotation_csv:
patch_size: 1500
patch_overlap: 0.05
min_score: 0.3
strategy: 'random'
# Number of images to select for annotation
n_images: 10

# Label studio
label-studio-url: "https://labelstudio.naturecast.org/"
label-studio-project: "Everglades Airplane Model"

# Debugging
force_run: True
skip_train: True
```

# Roadmap, Ideas, guiding principles, and wish list

Human review is here to stay. We need rapid model integration to create faster labeling environments specific to airborne biological use-cases. 

* Create an annotation platform that using existing tools (e.g. Label-studio) to detect and classify biological objects. We don't want to get bogged down in re-inventing annotation classes and UI.
* Pre-annotate imagery with existing model classes and varying levels of taxonomic detail ("duck" versus "White-winged Scoter")
* Batch labeling tools that operate on the flight or campaign level to interactively cluster and label detections grouped by distance in embedded space. In the clustering you can re-label groups of points with existing or new labels. Clicking on individual points takes you to images.
* Prompt-level engineering to find images with certain characteristics. "Find me all the images over water"
* Both detection level and image-level query to find detections similar to target.
* Pre-computed zoom levels based on detection density -> https://openaccess.thecvf.com/content_CVPRW_2020/papers/w11/Li_Density_Map_Guided_Object_Detection_in_Aerial_Images_CVPRW_2020_paper.pdf
* Nightly model training and re-labeling, re-clustering.
* Label propogation at the image level. If I click on one animal in a flock/herd, it should auto-update nearby objects.
* Label propogation at the annotation level, using SAM to go between points, boxes, and polygons.
* On new mission, draw a bounding box of geographic area, query the ebird API/map of life/Inaturalist to get abundance curve and filter species list.
* Double counting among images using keypoints

# Remaining questions
* Local Desktop installer? Especially for field researchers around the world? A stripped down version.
* How to learn from [AIDE](https://github.com/microsoft/aerial_wildlife_detection)? From [Scout](https://www.wildme.org/scout.html)? [Fathomnet Portal](https://fathomnet.org/fathomnet/#/), [SAM-geo](https://github.com/opengeos/segment-geospatial). Should we just merge with there? How do we promote community collaboration and avoid re-invention. 
* https://www.tator.io/? Another option.  
* BioCLIP foundation model -> https://arxiv.org/abs/2311.18803 versus more bespoke models? Engaging teams INat/Ebirds teams.
