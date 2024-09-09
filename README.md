# Airborne Field Guide

# Installation

```
conda create --name AirborneFieldGuide python=3.10
conda activate AirborneFieldGuide
pip install -r requirements.txt
```

# What does this repo do?

This repo is a development space for airborne machine learning model development using active learning and label-studio for rapid iteration. Unlabeled data is uploaded to label studio alongside existing model predictions. Human annotators correct and update these 'pre-annotations'. The data is then downloaded nightly to train a new model. 

## Contents

* A config .yml file that specifies task specific details such as label-studio project and active learning strategy

### Config

A general example of the yaml can be found of the .yml files. See pipeline.iterate for args. Any argument in the .yml will be passed to this function.

```
def config_pipeline(config, dask_client=None):
    iterate(dask_client=dask_client, **config)
```

### Pipeline

To run a pipeline, specify the name of the .yml file

```
run.py FWS
```

will run the pipeline with FWS_config.yml

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
