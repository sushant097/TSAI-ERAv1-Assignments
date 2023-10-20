import gradio as gr
import numpy as np
import torch
import clip
import cv2
from PIL import Image

# Function to perform object detection
def objectDetection(image, model):
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = model(image)
    croped_objects = result.crop(save=False)
    
    listOfObjects = []
    for obj in croped_objects:
        # listOfObjects.append(cv2.cvtColor(obj['im'], cv2.COLOR_BGR2RGB))
        listOfObjects.append(cv2.cvtColor(obj['im'], cv2.COLOR_BGR2RGB))

    detectedObjects = result.render()[0]

    return listOfObjects, detectedObjects

# Function to find objects in an image
def findObjects(listOfObjects, query, model, preprocess, device, N):
    objects = torch.stack([preprocess(Image.fromarray(im)) for im in listOfObjects]).to(device)

    with torch.no_grad():
        image_features = model.encode_image(objects)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        text_features = model.encode_text(clip.tokenize(query).to(device))
        text_features /= text_features.norm(dim=-1, keepdim=True)

    similarity = (text_features.cpu().numpy() @ image_features.cpu().numpy().T) * 100
    similarity = similarity[0]
    scores, images = similarity_top(similarity, listOfObjects, N=N)     

    return scores, images

# Function to find the top N similar objects
def similarity_top(similarity_list, listOfObjects, N):
    results = zip(range(len(similarity_list)), similarity_list)
    results = sorted(results, key=lambda x: x[1], reverse=True)
    images = []
    scores = []
    for index, score in results[:N]:
        scores.append(score)
        images.append(listOfObjects[index])

    return scores, images


# Function to load YOLOv5 and CLIP models
def get_model_session(OBJDETECTIONREPO, OBJDETECTIONMODEL, FINDERMODEL, DEVICE):
    models = []

    # Load YOLOv5 model
    yolo_model = torch.hub.load(OBJDETECTIONREPO, OBJDETECTIONMODEL)

    # Load CLIP model
    clip_model, preprocess = clip.load(FINDERMODEL, device=DEVICE)

    models.append(yolo_model)
    models.append(clip_model)
    models.append(preprocess)

    return models


# Function to perform the search and return output as a dictionary
def search_objects(image, query):
    listOfObjects, detectedObjects = objectDetection(image, models[0])
    scores, images = findObjects(listOfObjects, query, models[1], models[2], DEVICE, N)
    print(query)
    return_images = [(np.array(img), f'Score: {score}') for img, score in zip(images, scores)]
    return return_images



# Load models
OBJDETECTIONREPO = 'ultralytics/yolov5'
OBJDETECTIONMODEL = 'yolov5x6'  # Change to your desired YOLOv5 model
FINDERMODEL = 'ViT-B/32'  # Change to your desired CLIP model
DEVICE = 'cpu'  # Change to 'cuda' if you have a GPU
N = 4

models = get_model_session(OBJDETECTIONREPO, OBJDETECTIONMODEL, FINDERMODEL, DEVICE)


combined_iface = gr.Interface(
    fn=search_objects,
    inputs=[gr.Image(type='numpy', label='Image'), 'text'],
    outputs=['gallery'],
    title='🔍 Search Between the Objects',
    description='A simple interface to search between objects in an image using YOLO and CLIP models.',
)

combined_iface.launch()
