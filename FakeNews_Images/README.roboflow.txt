
Fake News Image Classifier - v2 2023-05-18 9:11am
==============================

This dataset was exported via roboflow.com on May 18, 2023 at 1:11 AM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 2068 images.
Fake-news-images are annotated in folder format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 640x640 (Stretch)

The following augmentation was applied to create 3 versions of each source image:
* Random rotation of between -20 and +20 degrees
* Random brigthness adjustment of between -37 and +37 percent
* Random exposure adjustment of between -25 and +25 percent
* Salt and pepper noise was applied to 3 percent of pixels


