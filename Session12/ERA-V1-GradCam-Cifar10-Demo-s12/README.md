---
title: ERA V1 GradCam Cifar10 Demo
emoji: 📊
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: 3.39.0
app_file: app.py
pinned: false
license: apache-2.0
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference


# Demo - Cifar10 - Custom Resnet 18 in PyTorch Lightning
## Gradio Interface - Visualizing GradCAM 

## Two Demos are created using tabbedinterface

## Demo 1 - Inference and GradCAM images


### Input

* First - Options whether to show GradCAM images

* Second - To see Number of missclassified images 

* Third - Customize the opacity level of the images displayed

* Fourth - How many top classes are to be shown (max 10)

### Output

Specified GradCAM images displayed with top predicted classes 

## Demo 2 - Misclassified Images

### Input

How many misclassified images the user wants to visualize (max 10)


### Output

Specified number of misclassified images are displayed with actual and predicted classes

#### Additionally the app allow users to upload new images
#### 10 example images are also provided to the user to try the app

## Github Link
Full Code: https://github.com/sushant097/TSAI-ERAv1-Assignments/tree/master/Session12
