# Airborne Field Guide
A deepforest object detection active learning tool

# Installation

```
conda env create -f=environment.yml
```

# Ideas, guiding principles, and wish list

Human review is here to stay. We need rapid model integration to create faster labeling environments specific to airborne biological use-cases. 

* Create an annotation platform that using existing tools (e.g. Label-studio) to detect and classify biological objects
* Pre-annotate imagery with existing model classes and varying levels of taxonomic detail ("duck" versus "White-winged Scoter")
* Batch labeling tools that operate on the flight or campaign level to interactively cluster and label detections grouped by distance in embedded space. In the clustering you can re-label groups of points with existing or new labels. Clicking on individual points takes you to images.
* Prompt-level engineering to find images with certain characteristics. "Find me all the images over water"
* Pre-computed zoom levels based on detection density -> https://openaccess.thecvf.com/content_CVPRW_2020/papers/w11/Li_Density_Map_Guided_Object_Detection_in_Aerial_Images_CVPRW_2020_paper.pdf
* Nightly model training and re-labeling, re-clustering
* Label propogation at the image level. If I click on one bird in a flock, it should auto-update nearby birds.
* On new mission, draw a bounding box of geographic area, query the ebird API/map of life/Inaturalist to get abundance curve and filter species list.

# Remaining questions
* Local installer?
* How to learn from AIDE? Should we just merge with there?
* https://www.tator.io/? Another option.
* BioCLIP foundation model -> https://arxiv.org/abs/2311.18803
